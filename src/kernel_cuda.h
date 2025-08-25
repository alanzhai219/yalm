#pragma once
#include "config.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#endif

/* CUDA KERNEL */
#ifdef USE_CUDA

__device__
float matmul_row(const float* row, const float* x, int offset, int dim);

__device__
float matmul_row(const half* row, const float* x, int offset, int dim);

template <typename T>
__global__
void matmul_wide(const T* A, const float* x, int n, int d, float* out);

template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d);

__global__
void rmsnorm_cuda(const float* x, const float* weight, int size, float eps, float* out);

__device__
inline void rope(const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, float* out);

__device__
inline void rope(const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out);

__device__
inline void rope(const half* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out);

void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads);

template <typename T>
void ffn_cuda(
  float* xout, float* x, 
  T* w1, T* w2, T* w3, 
  int hidden_dim, int dim,
  ActivationType act
);

// template <typename T>
// __global__
// void fused_qkv_matmul_clip(
//   const T* wq,      // (q_dim, dim)
//   const T* wk,      // (kv_dim, dim)
//   const T* wv,      // (kv_dim, dim)
//   const float* x,   // (dim,)
//   int dim,          // input dimension
//   int q_dim,        // n_heads * head_dim
//   int kv_dim,       // n_kv_heads * head_dim
//   float qkv_clip,   // clipping value
//   float* q_out,     // (q_dim,)
//   float* k_out,     // (kv_dim,)
//   float* v_out      // (kv_dim,)
// );

template <typename T>
__global__
void fused_matmul_add_residuals(const T* A, const float* x, int n, int d, float* out);

__global__
void copy_embedding_float(const float* token_embedding_table, int dim, int token, float* out);

__global__
void fused_rope_and_cache_update(
  const float* q,         // (n_heads * head_dim,)
  const float* k,         // (n_kv_heads * head_dim,)
  const float* v,         // (n_kv_heads * head_dim,)
  int head_dim,          
  int n_heads,
  int n_kv_heads,
  int pos,               // current position
  int kv_pos,           // position in KV cache
  float theta,          // RoPE theta parameter
  int rotary_dim,       // how many dimensions to rotate
  float* q_out,         // (n_heads * head_dim,)
  half* kb,            // (max_seq_len, n_kv_heads, head_dim)
  half* vb            // (max_seq_len, n_kv_heads, head_dim)
);

template <typename T, ActivationType A>
__global__
void fused_ffn_w1_w3_glu_act(
  const T* w1,        // (hidden_dim, dim)
  const T* w3,        // (hidden_dim, dim)
  const float* x,     // (dim,)
  int dim,           
  int hidden_dim,
  float* out         // (hidden_dim,)
);

__global__
void attn_dot(
  const half* kb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* q,   // (n_heads, head_dim)
  int head_dim, 
  int kv_len, 
  int max_seq_len, 
  int n_heads, 
  int n_kv_heads,
  float* out        // (n_heads, kv_len)
);

__global__
void attn_softmax(
  const float* att, 
  int seq_len, 
  int max_seq_len, 
  int n_heads, 
  float* out
);

__global__
void att_mix(
  const half* vb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
); 

__global__
void rotate_sink_tokens(
  half* kb, 
  int kv_sink, 				// number of attention sinks
  int kv_dim, 				// size of each entry (all concatenated heads) in KV cache
  int head_dim,
  float theta, 				// RoPE theta parameter
  int rotary_dim			// how many dimensions to rotate
);

__global__
void copy_embedding_half(const half* token_embedding_table, int dim, int token, float* out);
#endif

/* SYCL KERNEL */
#ifdef USE_SYCL
#endif
