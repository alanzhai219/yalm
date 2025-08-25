#include <assert.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include "immintrin.h"
#include "f16cintrin.h"
#include "kernel_cpu.h"
#include "utils.h"

void matmul_cpu(float* xout, float* x, float* w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    xout[i] = val;
  }
}

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
    __m256 sumlo = _mm256_setzero_ps();
    __m256 sumhi = _mm256_setzero_ps();
    for (int j = 0; j < n; j+=16) {
      // Extract the next set of 16 float16 weights from `w` and store them
      // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
      __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
      __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
      __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
      __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
      __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
      // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
      __m256 xveclo = _mm256_loadu_ps(&x[j]);
      __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
      // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
      sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
      sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
    }
    // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
    __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
    __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
      _mm256_extractf128_ps(sum8, 0), 
      _mm256_extractf128_ps(sum8, 1)
    );
    __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
    xout[i] = _mm_cvtss_f32(sum1);
  }
#else
  assert(false && "float16 not supported on this platform");
#endif
}

void moe_gate_cpu(float* moe_weights, int* active_experts, float* x, int n_experts, int n_active_experts) {
  // Set moe_weights[:n_active_experts] to the weights of the top K experts.
  // Set active_experts[:n_active_experts] to the indices of the top K experts.

  // get the max weight for later softmax computation
  float max_val = -FLT_MAX;
  for (int j = 0; j < n_experts; ++j) {
    if (x[j] > max_val) {
      max_val = x[j];
    }
  }
  
  // top k
  uint64_t mask = 0;
  float wsum = 0.0f;
  for (int k = 0; k < n_active_experts; ++k) {
    int best = -1;
    for (int j = 0; j < n_experts; ++j) {
      if ((mask & (1ull << j)) == 0 && (best == -1 || x[j] > x[best])) {
        best = j;
      }
    }

    active_experts[k] = best;
    wsum += expf(x[active_experts[k]] - max_val);
    mask |= 1ull << best;
  }

  // normalize top k weights to obtain the softmax result
  for (int k = 0; k < n_active_experts; ++k) {
    moe_weights[k] = expf(x[active_experts[k]] - max_val) / wsum;
  }
}

void rmsnorm_cpu(float* o, float* x, float* weight, int size, float eps) {
  float rms = 0.0f;
  for (int i = 0; i < size; ++i) {
    rms += x[i] * x[i];
  }
  rms = sqrtf(rms / size + eps);
  float scale = 1.0f / rms;
  for (int i = 0; i < size; ++i) {
    o[i] = x[i] * scale * weight[i];
  }
}

void layernorm_cpu(float* o, float* x, float* weight, float* bias, int size, float eps) {
  float mean = 0.0f;
  for (int i = 0; i < size; ++i) {
    mean += x[i];
  }
  mean /= size;
  float var = 0.0f;
  for (int i = 0; i < size; ++i) {
    var += (x[i] - mean) * (x[i] - mean);
  }
  var /= size;
  float scale = 1.0f / sqrtf(var + eps);
  if (bias) {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
  } else {
    for (int i = 0; i < size; ++i) {
      o[i] = (x[i] - mean) * scale * weight[i];
    }
  }
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
void softmax_cpu(float* o, float* x, int size) {
  float score_max = -FLT_MAX;
  for (int i = 0; i < size; ++i) {
    if (x[i] > score_max) {
      score_max = x[i];
    }
  }
  float score_sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    o[i] = expf(x[i] - score_max);
    score_sum += o[i];
  }
  for (int i = 0; i < size; ++i) {
    o[i] /= score_sum;
  }
}

float gelu_cpu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

float silu_cpu(float x) {
  return x / (1.0f + expf(-x));
}

float clip_cpu(float x, float v) {
  return x < -v ? -v : (x > v ? v : x);
}

// TODO annotate me
void rope_cpu(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

// Compute next value in a sequence for a single causal self-attention head.
void attn_cpu(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * half_to_float(kh[t * kv_stride + i]);
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax_cpu(atth, atth, kv_len);

  // mix values with attention weights
  for (int i = 0; i < head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * half_to_float(vh[t * kv_stride + i]);
    }
    xout[i] = vi;
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn_cpu(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, n_kv_heads, kv_len
    );
  }
}

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul_cpu(hb, x, w1, dim, hidden_dim);
  matmul_cpu(hb2, x, w3, dim, hidden_dim);
  switch (act) {
    case ActivationType::GELU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = gelu_cpu(hb[i]) * hb2[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < hidden_dim; ++i) {
        hb[i] = silu_cpu(hb[i]) * hb2[i];
      }
      break;
    }
  }

  matmul_cpu(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}
