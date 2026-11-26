# In a Brutus host module with the experimental flake as inputs.inkling:
#
#   let recipe = import "${inputs.inkling}/experiments/inkling-b70/brutus-inkling2.nix" {
#     runtime = inputs.inkling.packages.x86_64-linux.vllm-xpu-inkling2-kvarn;
#   }; in { imports = [ recipe.brutusModule ]; }
#
# With the ordinary vllm-xpu NixOS module, use the same recipe and set
# services.vllm-xpu.instances.chat = recipe.instance in the host configuration.
{
  runtime,
  maxModelLen ? 262144,
  gpuMemoryUtilization ? 0.95,
  flashPrefill ? true,
}:
let
  revision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
  instance = {
    enable = true;
    package = runtime;
    model = "thinkingmachines/Inkling-2-Mini";
    inherit revision maxModelLen gpuMemoryUtilization;
    servedName = "sunny-chat";
    dtype = "bfloat16";
    kvCacheDtype = "fp8";
    maxNumSeqs = 4;
    reasoningParser = "inkling";
    enableAutoToolChoice = true;
    toolCallParser = "inkling";
    limitMmPerPrompt.image = 2;
    speculativeConfig = {
      method = "dspark";
      num_speculative_tokens = 7;
      draft_sample_method = "probabilistic";
      # The DSpark config resolves its bundled draft revision independently.
      inherit revision;
    };
    extraEnvironment = {
      HF_HUB_OFFLINE = "1";
      INK2_CUTLASS_MOE = "1";
      INK2_SPARSE_CONV_HISTORY = "1";
      INK2_HYBRID_PREFIX_SDPA = "1";
      INK2_KVARN_K4V4 = "1";
      INK2_KVARN_FLASH_PREFILL = if flashPrefill then "1" else "0";
      VLLM_KV_CACHE_LAYOUT = "BLNHC";
    };
    extraArgs = [
      "--block-size"
      "128"
      "--max-num-batched-tokens"
      "2048"
      "--no-async-scheduling"
      "--compilation-config"
      (builtins.toJSON {
        mode = 0;
        cudagraph_mode = "FULL_DECODE_ONLY";
      })
      "--cudagraph-metrics"
      "--enable-extensible-kv-cache"
      "--enable-prefix-caching"
      "--kv-transfer-config"
      (builtins.toJSON {
        kv_connector = "OffloadingConnector";
        kv_role = "kv_both";
        kv_connector_extra_config = {
          spec_name = "CPUOffloadingSpec";
          cpu_bytes_to_use = 8 * 1024 * 1024 * 1024;
          gpu_cache_policy = "active_only";
          offload_prompt_only = false;
        };
      })
    ];
  };
in
{
  inherit instance;
  # Brutus derives the concrete unit and Bifrost catalog from local-inference.
  # Replace the selected backend as a unit so its Qwen-only options cannot merge.
  brutusModule =
    { lib, ... }:
    {
      services.local-inference.models.chat = {
        backendConfig = lib.mkForce { vllm-xpu = instance; };
        chatTemplate = lib.mkForce null;
        maxInputTokens = lib.mkForce maxModelLen;
        supportedReasoningEfforts = lib.mkForce [
          "none"
          "minimal"
          "low"
          "medium"
          "high"
          "xhigh"
          "max"
        ];
      };
    };
}
