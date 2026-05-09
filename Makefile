KERNELS_SRC ?=
BUILD_DIR ?= build-dev
JOBS ?= 6

ifeq ($(KERNELS_SRC),)
$(error KERNELS_SRC not set; pass KERNELS_SRC=/path/to/vllm-xpu-kernels)
endif

ifeq ($(VLLM_CUTLASS_SRC_DIR),)
$(error VLLM_CUTLASS_SRC_DIR not set; enter via 'nix develop .#attn-dev')
endif

# Mirror nix/vllm-xpu-attn-link.nix cmakeFlags so dev and prod stay aligned.
CMAKE_FLAGS = \
  -G Ninja \
  -S $(KERNELS_SRC) \
  -B $(BUILD_DIR) \
  -DVLLM_XPU_LIBS_ONLY=ON \
  -DVLLM_PYTHON_EXECUTABLE=$(shell command -v python3) \
  -DVLLM_CUTLASS_SRC_DIR=$(VLLM_CUTLASS_SRC_DIR) \
  -DCMAKE_BUILD_TYPE=Release \
  -DXE2_AOT_DEVICES=bmg \
  -DBUILD_SYCL_TLA_KERNELS=ON \
  -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
  -DBASIC_KERNELS_ENABLED=OFF \
  -DFA2_KERNELS_ENABLED=ON \
  -DMOE_KERNELS_ENABLED=OFF \
  -DGDN_KERNELS_ENABLED=OFF \
  -DMQA_LOGITS_KERNELS_ENABLED=OFF \
  -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
  -DXPUMEM_ALLOCATOR_ENABLED=OFF

ATTN_LIB = $(BUILD_DIR)/csrc/xpu/attn/xe_2/libattn_kernels_xe_2.so

.PHONY: dev-attn dev-attn-configure dev-attn-clean

dev-attn-configure: $(BUILD_DIR)/build.ninja

$(BUILD_DIR)/build.ninja:
	cmake $(CMAKE_FLAGS)

dev-attn: dev-attn-configure
	cmake --build $(BUILD_DIR) --target attn_kernels_xe_2 -- -j$(JOBS)
	@echo
	@echo "Built: $(ATTN_LIB)"
	@echo "Use it from a Python session via:"
	@echo "    export VLLM_XPU_DEV_LIB_DIR=$$PWD/$(BUILD_DIR)/csrc/xpu/attn/xe_2"
	@echo "    python -c 'import vllm_xpu_kernels'"

dev-attn-clean:
	rm -rf $(BUILD_DIR)
