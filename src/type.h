#pragma once
#include <cstdint>

using f16_t = uint16_t;

enum class ActivationType {
  GELU,
  SILU,
};

enum class LayerNormType {
  RMSNorm,
};

enum class Device {
  CPU,
  CUDA,
};

enum class InferenceMode {
  HYDRATE_KV_CACHE, // only hydrate the KV cache and don't compute output logits
  OUTPUT_LOGITS // set InferenceState logits to logits for the next token
};

enum class DType {
  UNKNOWN = 0,
  F32,
  F16,
  BF16,
  F8E5M2,
  F8E4M3,
  I32,
  I16,
  I8,
  U8,
};
