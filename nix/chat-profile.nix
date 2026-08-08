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
