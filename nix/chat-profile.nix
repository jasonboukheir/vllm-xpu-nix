{
  chat = {
    model = "shawnw3i/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP";
    revision = "ed099273cf30ad72a88116d759856f147b7bcbff";
    servedName = "qwen3.6-27b";
    dtype = "bfloat16";
    quantization = "auto_awq";
    kvCacheDtype = "auto";
    maxModelLen = 65536;
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
    languageModelOnly = false;
    limitMmPerPrompt = {
      image = 2;
      video = 0;
    };
    extraArgs = [
      "--override-generation-config"
      (builtins.toJSON {
        temperature = 1.0;
        top_k = 20;
        top_p = 0.95;
      })
    ];
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
