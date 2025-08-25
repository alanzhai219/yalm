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

std::string dtype_to_string(DType dtype);

DType string_to_dtype(const std::string& dtype_str);

size_t dtype_size(DType dtype);