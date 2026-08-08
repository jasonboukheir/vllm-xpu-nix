let
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
    cudagraphCaptureSizes = [3 6];
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
in {
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
      extraArgs = chat.extraArgs ++ ["--no-enable-prefix-caching"];
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
      extraArgs = chat.extraArgs ++ ["--no-enable-prefix-caching"];
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
      extraArgs = chat.extraArgs ++ ["--no-enable-prefix-caching"];
    };

  kvarnMtpEagerK4V4 = (builtins.removeAttrs chat ["cudagraphCaptureSizes"]) // {
    servedName = "qwen3.6-27b-kvarn-mtp-k4v4";
    kvCacheDtype = "kvarn_k4v4_g128";
    maxModelLen = 8192;
    enforceEager = true;
    enableXpuGraph = false;
    extraArgs = chat.extraArgs ++ ["--no-enable-prefix-caching"];
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
    };

  kvarnGraphK4V4 =
    (builtins.removeAttrs chat ["speculativeConfig"])
    // {
      servedName = "qwen3.6-27b-kvarn-graph-k4v4";
      kvCacheDtype = "kvarn_k4v4_g128";
      maxModelLen = 8192;
      extraArgs = chat.extraArgs ++ ["--no-enable-prefix-caching"];
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
    extraArgs = ["--trust-remote-code"];
  };
}
