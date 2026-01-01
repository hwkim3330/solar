# Model Architecture Comparison

Solar Open 100B와 다른 주요 LLM 모델들의 아키텍처 비교

## Quick Comparison Table

| Model | Total Params | Active Params | Type | Experts | Context | GQA |
|-------|-------------|---------------|------|---------|---------|-----|
| **Solar Open 100B** | 102.6B | 12B | MoE | 129 (1+128) | 128K | Yes (8 KV) |
| Mixtral 8x22B | 141B | 39B | MoE | 8 | 65K | Yes |
| DeepSeek-V3 | 671B | 37B | MoE | 257 (1+256) | 128K | Yes |
| Qwen2.5-72B | 72B | 72B | Dense | N/A | 128K | Yes |
| Llama 3.1 405B | 405B | 405B | Dense | N/A | 128K | Yes |
| GPT-4 (est.) | ~1.8T | ~220B | MoE | 16 | 128K | Unknown |

## Detailed Architecture Comparison

### Solar Open 100B
```
Architecture:     MoE with Shared Expert
Hidden Size:      4096
Layers:           48
Attention:        64 heads, 8 KV heads (GQA 8:1)
Experts:          1 shared + 128 routed (top-8)
FFN Size:         10240 (shared), 1280 (routed)
Vocab:            196,608
Rope Theta:       1,000,000
Active Params:    ~12B per token
```

### Mixtral 8x22B
```
Architecture:     Standard MoE
Hidden Size:      6144
Layers:           56
Attention:        48 heads, 8 KV heads (GQA 6:1)
Experts:          8 (top-2)
FFN Size:         16384 per expert
Vocab:            32,000
Rope Theta:       1,000,000
Active Params:    ~39B per token
```

### DeepSeek-V3
```
Architecture:     MoE with Shared Expert + Auxiliary Loss
Hidden Size:      7168
Layers:           61
Attention:        Multi-head Latent Attention (MLA)
Experts:          1 shared + 256 routed (top-8)
FFN Size:         Variable (fine-grained experts)
Vocab:            129,280
Active Params:    ~37B per token
```

### Qwen2.5-72B
```
Architecture:     Dense Transformer
Hidden Size:      8192
Layers:           80
Attention:        64 heads, 8 KV heads (GQA 8:1)
FFN Size:         29568
Vocab:            152,064
Rope Theta:       1,000,000
Active Params:    72B (all)
```

## Expert Routing Strategies

### 1. Solar Open 100B: Shared + Top-K Routing
```
Input Token
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
[Shared Expert]                    [Router Network]
  (Always On)                           │
    │                                   │
    │                          ┌────────┼────────┐
    │                          ▼        ▼        ▼
    │                      [Expert 1][Expert 2]...[Expert 8]
    │                          │        │        │   (Top-8)
    │                          └────────┼────────┘
    │                                   │
    └───────────────┬───────────────────┘
                    │
                    ▼
              [Weighted Sum]
```

### 2. Mixtral: Simple Top-K Routing
```
Input Token
    │
    ▼
[Router Network]
    │
    ├─────────┐
    ▼         ▼
[Expert 1] [Expert 2]  (Top-2 of 8)
    │         │
    └────┬────┘
         │
         ▼
   [Weighted Sum]
```

### 3. DeepSeek-V3: Fine-Grained + Shared
```
Input Token
    │
    ├──────────────────────────────────────┐
    │                                      │
    ▼                                      ▼
[Shared Expert]                    [Auxiliary-Free Router]
  (Always On)                           │
    │                                   │
    │                    ┌──────────────┼──────────────┐
    │                    ▼              ▼              ▼
    │              [Fine-grained Experts × 256]
    │                    (Top-8 selection)
    │                    │
    └───────────────┬────┘
                    │
                    ▼
              [Sum with Balancing]
```

## Attention Mechanisms

### Grouped Query Attention (GQA)

Solar Open 100B uses GQA with 8:1 ratio:
- 64 Query heads
- 8 Key-Value heads
- Each KV head serves 8 Query heads

**Memory Savings:**
```
Standard MHA KV Cache:  64 heads × 128 dim = 8192 per position
GQA (8 KV heads):       8 heads × 128 dim = 1024 per position
Savings:                8× reduction
```

### Multi-head Latent Attention (MLA) - DeepSeek-V3

DeepSeek-V3 uses a novel MLA mechanism:
- Compresses KV projections
- Further reduces memory
- Similar efficiency to GQA but different approach

## FFN Design Comparison

### Solar Open 100B
```python
# Shared Expert (larger)
shared_output = down_proj(
    silu(gate_proj(x)) * up_proj(x)
)  # SwiGLU, intermediate=10240

# Routed Experts (smaller, many)
for expert in selected_experts:
    expert_output = expert.down_proj(
        silu(expert.gate_proj(x)) * expert.up_proj(x)
    )  # SwiGLU, intermediate=1280
```

### Mixtral 8x22B
```python
# All experts same size
for expert in top2_experts:
    expert_output = expert.down_proj(
        silu(expert.gate_proj(x)) * expert.up_proj(x)
    )  # SwiGLU, intermediate=16384
```

## Training Data Comparison

| Model | Training Tokens | Data Composition |
|-------|----------------|------------------|
| Solar Open 100B | 19.7T | Multilingual, Korean-focused |
| Mixtral 8x22B | Unknown | Web, code, multilingual |
| DeepSeek-V3 | 14.8T | Web, code, math |
| Llama 3.1 405B | 15T+ | Web, code, multilingual |

## Efficiency Analysis

### Compute per Token (Active Params)

```
Solar Open 100B:    12B active    (11.7% of total)
Mixtral 8x22B:      39B active    (27.7% of total)
DeepSeek-V3:        37B active    (5.5% of total)
Qwen2.5-72B:        72B active    (100% - dense)
```

### Inference Speed (Theoretical, same hardware)

Assuming same batch size and hardware:
```
Model               Active Params    Relative Speed
Solar Open 100B     12B              1.0x (baseline)
DeepSeek-V3         37B              ~0.32x
Mixtral 8x22B       39B              ~0.31x
Qwen2.5-72B         72B              ~0.17x
```

## Key Innovations

### Solar Open 100B
1. **Shared Expert Design**: 1 large shared + 128 small routed
2. **High Expert Count**: 129 total experts for specialization
3. **Aggressive MoE Sparsity**: Only 12B/102.6B active (11.7%)
4. **Extended RoPE**: theta=1M for 128K context

### DeepSeek-V3
1. **Auxiliary-Free Load Balancing**: No auxiliary loss
2. **Multi-Token Prediction**: Predicts multiple tokens
3. **FP8 Training**: Lower precision training

### Mixtral
1. **Simplicity**: Standard top-2 routing
2. **Proven Architecture**: Well-tested design
3. **Open Weights**: Fully open source

## Conclusion

Solar Open 100B는 다음 특징으로 차별화됩니다:

1. **최적화된 MoE 설계**: Shared + Routed 구조로 효율성 극대화
2. **높은 희소성**: 11.7%만 활성화로 빠른 추론
3. **한국어 최적화**: 한국어 성능에 특화
4. **대규모 학습**: 19.7T 토큰으로 광범위한 지식

가장 효율적인 MoE 설계 중 하나로, Dense 모델 대비 ~8x 빠른 추론이 가능합니다.
