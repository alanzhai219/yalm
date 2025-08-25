#pragma once
#include <iostream>
#include <string>
#include <assert.h>
#include <immintrin.h>
#include <f16cintrin.h>

#include "type.h"

#if defined(__AVX2__) && defined(__F16C__)
inline float half_to_float(f16_t x) {
  return _cvtsh_ss(x);
}
inline f16_t float_to_half(float x) {
  return _cvtss_sh(x, 0);
}
#else
inline float half_to_float(f16_t x) {
  assert(false && "float16 not supported on this platform");
  return 0.0f;
}
inline f16_t float_to_half(float x) {
  assert(false && "float16 not supported on this platform");
  return 0;
}
#endif

inline std::string dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::F8E5M2: return "F8_E5M2";
    case DType::F8E4M3: return "F8_E4M3";
    case DType::I32: return "I32";
    case DType::I16: return "I16";
    case DType::I8: return "I8";
    case DType::U8: return "U8";
    case DType::UNKNOWN: return "UNKNOWN";
    default: return "UNKNOWN";
  }
}

inline DType string_to_dtype(const std::string& dtype_str) {
  if (dtype_str == "F32") {
    return DType::F32;
  } else if (dtype_str == "F16") {
    return DType::F16;
  } else if (dtype_str == "BF16") {
    return DType::BF16;
  } else if (dtype_str == "F8_E5M2") {
    return DType::F8E5M2;
  } else if (dtype_str == "F8_E4M3") {
    return DType::F8E4M3;
  } else if (dtype_str == "I32") {
    return DType::I32;
  } else if (dtype_str == "I16") {
    return DType::I16;
  } else if (dtype_str == "I8") {
    return DType::I8;
  } else if (dtype_str == "U8") {
    return DType::U8;
  } else {
    std::cerr << "bad dtype" << std::endl;
    return DType::UNKNOWN;
  }
}

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::F8E5M2: return 1;
    case DType::F8E4M3: return 1;
    case DType::I32: return 4;
    case DType::I16: return 2;
    case DType::I8: return 1;
    case DType::U8: return 1;
    default: return 0;
  }
}
