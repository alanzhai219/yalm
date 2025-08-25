#pragma once
#include "config.h"

/* CPU KERNEL */
void matmul_cpu(float* xout, float* x, float* w, int n, int d);

void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d);

void moe_gate_cpu(float* moe_weights, int* active_experts, float* x, int n_experts, int n_active_experts);

void rmsnorm_cpu(float* o, float* x, float* weight, int size, float eps);

[[maybe_unused]]
void layernorm_cpu(float* o, float* x, float* weight, float* bias, int size, float eps);

void softmax_cpu(float* o, float* x, int size);

void rope_cpu(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim);

void attn_cpu(float* xout, float* atth, float* qh, f16_t* kh, f16_t* vh, int head_dim, int n_kv_heads, int kv_len);

void mha_cpu(float* xout,
             float* att, f16_t* kb, f16_t* vb, float* q, int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads);

void ffn_cpu(float* x_out, float* x, float* w1, float* w2, float* w3, int hidden_dim, int dim, ActivationType act);

float gelu_cpu(float x);

float silu_cpu(float x);

float clip_cpu(float x, float v);

/* CUDA KERNEL */
#ifdef USE_CUDA
#endif

/* SYCL KERNEL */
#ifdef USE_SYCL
#endif
