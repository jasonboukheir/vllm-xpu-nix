let
  kvarnRepresentationAttrs = [
    "kvCacheDtype"
    "kvarnDpasSafe"
    "kvarnFusedVerifyMinBlocks"
    "kvarnNativeDecode"
    "kvarnNativeHadamardScatter"
    "kvarnNativeSplits"
    "kvarnSinkhornIters"
  ];

  chat = {
    model = "Lorbus/Qwen3.6-27B-int4-AutoRound";
    servedName = "qwen3.6-27b";
    dtype = "bfloat16";
    quantization = "inc";
    kvCacheDtype = "auto";
    maxModelLen = 114688;
    maxNumSeqs = 4;
    gpuMemoryUtilization = 0.90;
    speculativeConfig = {
      method = "mtp";
      num_speculative_tokens = 2;
    };
    enforceEager = false;
    enableXpuGraph = true;
    cudagraphCaptureSizes = [
      3
      6
    ];
    reasoningParser = "qwen3";
    enableAutoToolChoice = true;
    toolCallParser = "qwen3_xml";
    languageModelOnly = true;
    extraArgs = [
      "--override-generation-config"
      (builtins.toJSON {
        temperature = 1.0;
        top_k = 20;
        top_p = 0.95;
      })
    ];
  };
in
rec {
  inherit chat;

  # Paired accuracy baseline: identical AutoRound weights and native BF16 GDN
  # state, with only the full-attention KV representation differing from the
  # KVarN eager profiles below.
  bf16KvEager =
    (builtins.removeAttrs chat [
      "speculativeConfig"
      "cudagraphCaptureSizes"
    ])
    // {
      servedName = "qwen3.6-27b-bf16-kv-eager";
      kvCacheDtype = "auto";
      maxModelLen = 8192;
      enforceEager = true;
      enableXpuGraph = false;
      extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
    };

  # Matched graph-mode control for the compact KVarN performance gate.
  bf16KvGraph = (builtins.removeAttrs chat [ "speculativeConfig" ]) // {
    servedName = "qwen3.6-27b-bf16-kv-graph";
    kvCacheDtype = "auto";
    maxModelLen = 8192;
    cudagraphCaptureSizes = [ 4 ];
    extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
  };

  # Matched native-KV controls for the two-token MTP correctness gate. These
  # deliberately retain the production qlen=3 graph sizes and prefix-cache
  # setting; only speculative decoding differs between the pair.
  bf16KvMtpGraph = chat // {
    servedName = "qwen3.6-27b-bf16-kv-mtp-graph";
    kvCacheDtype = "auto";
    maxModelLen = 8192;
    cudagraphCaptureSizes = chat.cudagraphCaptureSizes;
    # Keep the scheduler matched to the compact KVarN MTP candidate. KVarN
    # currently requires exact host lengths, which async speculative scheduling
    # does not expose without a device-to-host synchronization.
    asyncScheduling = false;
  };

  # Final direct-vLLM production control. Brutus runs the candidate below at
  # maxModelLen=114688; this control deliberately retains that context
  # envelope, MTP2, B4 admission, graph sizes, scheduler, model, sampling, and
  # API settings. Sequential startup is expected because both consume the same
  # XPU and memory budget.
  brutusBf16KvMtpGraph = chat // {
    servedName = "qwen3.6-27b-kvarn-compact-mtp-graph-k4v4";
    kvCacheDtype = "auto";
    asyncScheduling = false;
  };

  bf16KvPrefixGraph =
    (builtins.removeAttrs bf16KvMtpGraph [ "speculativeConfig" ])
    // {
      servedName = "qwen3.6-27b-bf16-kv-prefix-graph";
    };

  kvarnEagerK4V4 =
    (builtins.removeAttrs chat [
      "speculativeConfig"
      "cudagraphCaptureSizes"
    ])
    // {
      servedName = "qwen3.6-27b-kvarn-k4v4";
      kvCacheDtype = "kvarn_k4v4_g128";
      maxModelLen = 8192;
      enforceEager = true;
      enableXpuGraph = false;
      extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
    };

  kvarnCompactEagerK4V4 = kvarnEagerK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128_compact";
    kvarnDpasSafe = true;
  };

  # Handwritten Xe2 reader/producer, retained in a separate profile after its
  # full-vocabulary accuracy and matched serving gates passed.
  kvarnCompactNativeEagerK4V4 = kvarnCompactEagerK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-native-k4v4";
    kvarnNativeDecode = true;
    # The native kernel falls back to direct split-1 until there is at least
    # one 64-token tile per split, then uses split-16 for long-context decode.
    kvarnNativeSplits = 16;
    kvarnNativePersistentScratch = false;
    kvarnNativeHadamardScatter = true;
  };

  kvarnEagerK4V2 =
    (builtins.removeAttrs chat [
      "speculativeConfig"
      "cudagraphCaptureSizes"
    ])
    // {
      servedName = "qwen3.6-27b-kvarn-k4v2";
      kvCacheDtype = "kvarn_k4v2_g128";
      maxModelLen = 8192;
      enforceEager = true;
      enableXpuGraph = false;
      extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
    };

  kvarnMtpEagerK4V4 = (builtins.removeAttrs chat [ "cudagraphCaptureSizes" ]) // {
    servedName = "qwen3.6-27b-kvarn-mtp-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128";
    maxModelLen = 8192;
    enforceEager = true;
    enableXpuGraph = false;
    extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
  };

  # Compact production candidate with two-token MTP. Keep the native compact
  # producer, but use the Triton reader for both one-token drafter passes and
  # three-token verification; the optional native reader has a separate MTP
  # batch/graph gate.
  kvarnCompactMtpEagerK4V4 = kvarnCompactEagerK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-mtp-k4v4";
    speculativeConfig = chat.speculativeConfig;
    kvarnNativeHadamardScatter = true;
    # Materialize + FlashAttention is substantially faster for short and
    # medium cached histories. Switch to the native fused verifier only once
    # avoiding its O(context) materialization is expected to amortize the
    # higher per-query launch/reduction cost (64 KVarN blocks = 8192 tokens).
    kvarnFusedVerifyMinBlocks = 64;
  };

  kvarnPrefixEagerK4V4 =
    (builtins.removeAttrs chat [
      "speculativeConfig"
      "cudagraphCaptureSizes"
    ])
    // {
      servedName = "qwen3.6-27b-kvarn-prefix-eager-k4v4";
      kvCacheDtype = "kvarn_k4v4_g128";
      maxModelLen = 8192;
      enforceEager = true;
      enableXpuGraph = false;
      extraArgs = chat.extraArgs ++ [ "--enable-prefix-caching" ];
    };

  kvarnCompactPrefixEagerK4V4 = kvarnPrefixEagerK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-prefix-eager-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128_compact";
    kvarnDpasSafe = true;
  };

  kvarnGraphK4V4 = (builtins.removeAttrs chat [ "speculativeConfig" ]) // {
    servedName = "qwen3.6-27b-kvarn-graph-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128";
    maxModelLen = 8192;
    cudagraphCaptureSizes = [ 4 ];
    extraArgs = chat.extraArgs ++ [ "--no-enable-prefix-caching" ];
  };

  kvarnCompactGraphK4V4 = kvarnGraphK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-graph-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128_compact";
    kvarnDpasSafe = true;
  };

  kvarnCompactNativeGraphK4V4 = kvarnCompactGraphK4V4 // {
    servedName = "qwen3.6-27b-kvarn-compact-native-graph-k4v4";
    kvarnNativeDecode = true;
    kvarnNativeSplits = 16;
    kvarnNativePersistentScratch = false;
    kvarnNativeHadamardScatter = true;
  };

  kvarnCompactMtpGraphK4V4 = chat // {
    servedName = "qwen3.6-27b-kvarn-compact-mtp-graph-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128_compact";
    maxModelLen = 8192;
    kvarnDpasSafe = true;
    kvarnNativeDecode = true;
    # B12 (B4 x qlen3) reader/kernel winner; also preserves the best MTP
    # acceptance. Graph sizes [3, 6] cover B1/B2 verification. B3/B4 execute
    # this same retained path through intentional eager fallback so extra graph
    # captures do not consume the KV-cache VRAM budget.
    kvarnNativeSplits = 32;
    # Four iterations is the forced-decode-validated compact K4V4 setting.
    kvarnSinkhornIters = 4;
    cudagraphCaptureSizes = chat.cudagraphCaptureSizes;
    kvarnNativeHadamardScatter = true;
    # Keep the materialized FlashAttention verifier below the measured
    # crossover; forcing the fused verifier at 6K cut B4 throughput nearly in
    # half (73.41 -> 40.57 tok/s).
    kvarnFusedVerifyMinBlocks = 64;
    # KVarN makes host-side tile ownership/flush decisions from exact committed
    # lengths. Async speculative scheduling keeps rejection correction only on
    # device, forcing a D2H queue barrier in every drafter pass.
    asyncScheduling = false;
  };

  # Final direct-vLLM production candidate. Keep the public served name and
  # full Brutus context envelope identical to the BF16 control; only the KV
  # representation and representation-specific KVarN implementation knobs
  # differ.
  brutusKvarnCompactMtpGraphK4V4 =
    let
      candidate = kvarnCompactMtpGraphK4V4 // {
        servedName = brutusBf16KvMtpGraph.servedName;
        maxModelLen = brutusBf16KvMtpGraph.maxModelLen;
      };
    in
      assert
        builtins.removeAttrs candidate kvarnRepresentationAttrs
        == builtins.removeAttrs brutusBf16KvMtpGraph kvarnRepresentationAttrs;
      candidate;

  kvarnCompactPrefixGraphK4V4 =
    (builtins.removeAttrs kvarnCompactMtpGraphK4V4 [ "speculativeConfig" ])
    // {
      servedName = "qwen3.6-27b-kvarn-compact-prefix-graph-k4v4";
    };

  # Final integration gate: exercise all cache consumers together. Keep the
  # context bounded until the eager prefix and MTP state-content gates pass.
  kvarnMtpPrefixGraphK4V4 = chat // {
    servedName = "qwen3.6-27b-kvarn-mtp-prefix-graph-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128";
    maxModelLen = 8192;
  };

  embedding = {
    model = "jinaai/jina-embeddings-v5-text-nano-retrieval";
    servedName = "jina-embeddings-v5-nano";
    dtype = "bfloat16";
    runner = "pooling";
    maxModelLen = 8192;
    maxNumSeqs = 8;
    gpuMemoryUtilization = 0.05;
    enforceEager = true;
    extraArgs = [ "--trust-remote-code" ];
  };
}
