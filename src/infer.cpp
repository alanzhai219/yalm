#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "immintrin.h"
#include "f16cintrin.h"
#include "kernel.h"
#include "utils.h"

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cpu() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* x, size_t size) {
  std::vector<T> out(size);
  for (size_t i = 0; i < size; i++) {
    out[i] = x[i];
  }
  return out;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
template <typename T>
void Block::_block_cpu(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  assert(_config);
  const Config& c = *_config;

  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm_cpu(s.xb(), s.x(), rms_att_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  matmul_cpu(s.q(), s.xb(), wq<T>(), c.dim, q_dim);
  matmul_cpu(s.k(), s.xb(), wk<T>(), c.dim, kv_dim);
  matmul_cpu(s.v(), s.xb(), wv<T>(), c.dim, kv_dim);

  // some models require clipping qkv values
  for (int i = 0; i < q_dim; ++i) {
    s.q()[i] = clip_cpu(s.q()[i], c.qkv_clip);
  }
  for (int i = 0; i < kv_dim; ++i) {
    s.k()[i] = clip_cpu(s.k()[i], c.qkv_clip);
    s.v()[i] = clip_cpu(s.v()[i], c.qkv_clip);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope_cpu(s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  rope_cpu(s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  
  // key and value point to the kv cache
  f16_t* kb = key_cache();
  f16_t* vb = value_cache();
  // update kv cache
  for (int i = 0; i < kv_dim; ++i) {
    kb[kv_pos * kv_dim + i] = float_to_half(s.k()[i]);
    vb[kv_pos * kv_dim + i] = float_to_half(s.v()[i]);
  }

  // Sink tokens remain untouched while the rest of the KV cache is incrementally 
  // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
  // away from current timestep. Hence, each forward pass, rotate sink tokens 
  // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
  for (int r = 0; r < kv_sink; r++) {
    for (int i = 0; i < kv_dim; ++i) {
      s.k()[i] = half_to_float(kb[r * kv_dim + i]);
    }

    rope_cpu(s.k(), kv_dim, c.head_dim, 1, c.rope_theta, c.rotary_dim);

    for (int i = 0; i < kv_dim; i++) {
      kb[r * kv_dim + i] = float_to_half(s.k()[i]);
    }
  }

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < c.n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * c.head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn_cpu(s.xb2(h), s.att(h), s.q(h), kh, vh, c.head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul_cpu(s.hb(), s.xb2(), wo<T>(), q_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm_cpu(s.xb(), s.x(), rms_ffn_weight(), c.dim, c.norm_eps);
      break;
    }
  }

  if (c.n_experts > 0) {
    matmul_cpu(s.moe_weights(), s.xb(), moegate<T>(), c.dim, c.n_experts);
    moe_gate_cpu(s.active_experts_weights(), s.active_experts(), s.moe_weights(), c.n_experts, c.n_experts_active);
  } else {
    s.active_experts_weights()[0] = 1.0f;
    s.active_experts()[0] = 0;
  }

  for (int k = 0; k < (c.n_experts > 0 ? c.n_experts_active : 1); ++k) {
    int expert_index = s.active_experts()[k];
    int expert_size = c.dim * c.hidden_dim;
    // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
    // Note this is a feedforward with a GLU, not a simple MLP.
    matmul_cpu(s.hb(), s.xb(), w1<T>() + expert_index * expert_size, c.dim, c.hidden_dim);
    matmul_cpu(s.hb2(), s.xb(), w3<T>() + expert_index * expert_size, c.dim, c.hidden_dim);
    switch (c.act) {
      case ActivationType::GELU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = gelu_cpu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
      case ActivationType::SILU: {
        for (int i = 0; i < c.hidden_dim; ++i) {
          s.hb()[i] = silu_cpu(s.hb()[i]) * s.hb2()[i];
        }
        break;
      }
    }

    matmul_cpu(s.xb2(), s.hb(), w2<T>() + expert_index * expert_size, c.hidden_dim, c.dim);
    
    float expert_weight = s.active_experts_weights()[k];
    // residual connection back into x
    for (int i = 0; i < c.dim; ++i) {
      s.x()[i] += s.xb2()[i] * expert_weight;
    }
  }
}

template void Block::_block_cpu<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cpu<f16_t>(InferenceState&, int, int, int, int) const;

void Model::_copy_embedding(InferenceState& s, int token) {
  const Config& c = *config;
  switch (c.weight_dtype) {
    case DType::F32: {
      float* emb = static_cast<float*>(token_embedding_table);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case DType::F16: {
      f16_t* emb = static_cast<f16_t*>(token_embedding_table);
      for (int i = 0; i < c.dim; i+=1) {
        s.x()[i] = half_to_float(emb[token * c.dim + i]);
      }
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;

  // copy the token embedding into `x`
  _copy_embedding(s, token);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    return;
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      rmsnorm_cpu(s.x(), s.x(), rms_final_weight, c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  switch (c.weight_dtype) {
    case DType::F32: {
      matmul_cpu(s.logits(), s.x(), static_cast<float*>(wcls), c.dim, c.vocab_size);
      break;
    }
    case DType::F16: {
      matmul_cpu(s.logits(), s.x(), static_cast<f16_t*>(wcls), c.dim, c.vocab_size);
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}