import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file  # 用于加载模型权重
import sentencepiece as spm  # 分词器
import json

# RMSNorm 实现
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

# SwiGLU 激活函数
class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        gate = F.silu(self.w1(x))  # SiLU 激活
        x = self.w2(x) * gate
        return self.w3(x)

# 注意力机制（简化的 GQA + 滑动窗口）
class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, window_size=32768):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, T, head_dim]

        # 滑动窗口注意力（简化为全注意力，实际需限制窗口）
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.wo(out)

# Transformer 解码器层
class TransformerLayer(nn.Module):
    def __init__(self, dim=4096, num_heads=32, head_dim=128, ffn_dim=14336):
        super().__init__()
        self.attn = Attention(dim, num_heads, head_dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Mistral-7B 模型
class Mistral7B(nn.Module):
    def __init__(self, vocab_size=32000, dim=4096, num_layers=32, num_heads=32, head_dim=128, ffn_dim=14336):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerLayer(dim, num_heads, head_dim, ffn_dim) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)

# 分词器
def load_tokenizer(tokenizer_path):
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp


# 加载配置
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 加载权重
def load_mistral_weights(model, index_path, model_dir):
    with open(index_path, 'r') as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    state_dict = {}
    for key, file_name in weight_map.items():
        file_path = f"{model_dir}/{file_name}"
        shard = load_file(file_path)
        state_dict.update(shard)
    adjusted_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(adjusted_state_dict, strict=False)
    return model

# 分词器
class MistralTokenizer:
    def __init__(self, tokenizer_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text):
        if "[INST]" in text:
            text = text.replace("[INST]", "<s>[INST]").replace("[/INST]", "[/INST]</s>")
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode(tokens)

# 推理
@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, device='cuda'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    for _ in range(max_tokens):
        logits = model(input_ids)[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_id:
            break
    
    return tokenizer.decode(input_ids[0].cpu().numpy().tolist())

# 主程序
if __name__ == "__main__":
    # 加载配置
    model_dir = "/home/xiuchuan/xiuchuan/models/Mistral-7B-Instruct-v0.2"  # 替换为实际路径
    config_path = f"{model_dir}/config.json"
    config = load_config(config_path)
    
    # 初始化模型
    vocab_size = config.get("vocab_size", 32000)
    dim = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    head_dim = config.get("head_dim", dim // num_heads)
    ffn_dim = config.get("intermediate_size", 14336)
    model = Mistral7B(vocab_size, dim, num_layers, num_heads, head_dim, ffn_dim)
    
    # 加载权重
    index_path = f"{model_dir}/model.safetensors.index.json"
    model = load_mistral_weights(model, index_path, model_dir)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载分词器
    tokenizer_path = f"{model_dir}/tokenizer.model"
    tokenizer = MistralTokenizer(tokenizer_path)
    
    # 推理
    prompt = "<s>[INST] Explain Machine Learning in a nutshell. [/INST]"
    output = generate(model, tokenizer, prompt, max_tokens=100)
    print(output)