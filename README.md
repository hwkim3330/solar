# Solar Open 100B - Model Architecture Analysis

> Upstage의 Solar Open 100B 모델 구조 분석 레포지토리

[![Model](https://img.shields.io/badge/HuggingFace-Solar--Open--100B-yellow)](https://huggingface.co/upstage/Solar-Open-100B)
[![Paper](https://img.shields.io/badge/arXiv-2312.15166-red)](https://arxiv.org/abs/2312.15166)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

## Overview

Solar Open 100B는 Upstage에서 개발한 102.6B 파라미터의 대규모 언어 모델입니다. **Mixture-of-Experts (MoE)** 아키텍처를 채용하여 102B의 지식 깊이를 12B 활성 파라미터의 추론 속도로 제공합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Solar Open 100B                             │
├─────────────────────────────────────────────────────────────────┤
│  Total Parameters: 102.6B    Active Parameters: 12B per token   │
│  Pre-training: 19.7T tokens  Context Length: 128K tokens        │
│  Architecture: MoE (129 Experts = 128 Routed + 1 Shared)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Specifications

| Specification | Value |
|--------------|-------|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 102.6B |
| **Active Parameters** | 12B per token |
| **Hidden Size** | 4096 |
| **Number of Layers** | 48 |
| **Attention Heads** | 64 |
| **Key-Value Heads** | 8 (GQA) |
| **Head Dimension** | 128 |
| **Vocabulary Size** | 196,608 |
| **Context Length** | 128K tokens |
| **Pre-training Tokens** | 19.7 Trillion |
| **Training Hardware** | NVIDIA B200 GPUs |

---

## Architecture Deep Dive

### 1. Mixture-of-Experts (MoE) Architecture

Solar Open 100B는 129개의 Expert로 구성된 MoE 구조를 사용합니다:

```
                    ┌─────────────────┐
                    │   Input Token   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Router Network │
                    │   (Gating)      │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │   Shared    │   │  Routed     │   │  Routed     │
    │   Expert    │   │  Expert 1   │   │  Expert 8   │
    │  (Always)   │   │  (Top-8)    │   │  (Top-8)    │
    └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Weighted Sum   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Output      │
                    └─────────────────┘
```

### 2. Expert Configuration

```python
# Expert 구성
n_routed_experts = 128      # 라우팅되는 전문가 수
n_shared_experts = 1        # 항상 활성화되는 공유 전문가
num_experts_per_tok = 8     # 토큰당 선택되는 전문가 수

# 활성 파라미터 계산
# Shared Expert (1) + Top-8 Routed Experts = 9 Experts active
# 하지만 공유 전문가가 더 작아서 총 ~12B 활성 파라미터
```

### 3. Layer Structure (48 Layers)

```
┌──────────────────────────────────────────────────────────┐
│                    Transformer Layer                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │              Multi-Head Attention                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │ │
│  │  │  Query   │ │   Key    │ │  Value   │           │ │
│  │  │ (64 Heads)│ │(8 KV Heads)│(8 KV Heads)│         │ │
│  │  └──────────┘ └──────────┘ └──────────┘           │ │
│  │           ↓         ↓            ↓                 │ │
│  │      ┌─────────────────────────────────┐          │ │
│  │      │   Grouped Query Attention (GQA) │          │ │
│  │      │      Head Dim: 128              │          │ │
│  │      └─────────────────────────────────┘          │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 RMSNorm + Residual                  │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │              MoE Feed-Forward Layer                 │ │
│  │  ┌─────────────────────────────────────────────┐  │ │
│  │  │         Router (Softmax Gating)             │  │ │
│  │  └─────────────────────────────────────────────┘  │ │
│  │         ↓                         ↓               │ │
│  │  ┌───────────┐  ┌───────────────────────────┐    │ │
│  │  │  Shared   │  │    Routed Experts (Top-8) │    │ │
│  │  │  Expert   │  │  Expert_i: 4096→1280→4096 │    │ │
│  │  │4096→10240 │  │                           │    │ │
│  │  │  →4096    │  │                           │    │ │
│  │  └───────────┘  └───────────────────────────┘    │ │
│  └────────────────────────────────────────────────────┘ │
│                          ↓                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │                 RMSNorm + Residual                  │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          × 48 Layers
```

### 4. Grouped Query Attention (GQA)

```
Query Heads:      64 heads (full attention)
Key-Value Heads:   8 heads (grouped, shared across queries)

Memory Efficiency: 8× reduction in KV cache size
                   Enables 128K context length

┌─────────────────────────────────────────────────────────┐
│                  Grouped Query Attention                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Query: [batch, seq, 64, 128]                          │
│                    ↓                                     │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐            │
│   │Q1-8│Q9-16│...                   │Q57-64│            │
│   └──┬─┴──┬─┴────────────────────────┴──┬──┘            │
│      │    │                             │               │
│      ▼    ▼                             ▼               │
│   ┌────┐┌────┐                       ┌────┐             │
│   │ K1 ││ K2 │  ...                  │ K8 │             │
│   │ V1 ││ V2 │                       │ V8 │             │
│   └────┘└────┘                       └────┘             │
│                                                          │
│   8 Query heads share 1 KV head                         │
└─────────────────────────────────────────────────────────┘
```

### 5. Rotary Position Embedding (RoPE)

```python
# RoPE Configuration
max_position_embeddings = 131072  # 128K context
rope_theta = 1000000              # Extended theta for long context
partial_rotary_factor = 1.0       # Full rotary embedding

# RoPE 수식
# R(θ) = [cos(mθ), -sin(mθ)]
#        [sin(mθ),  cos(mθ)]
#
# 여기서 m = position, θ = θ_base^(-2i/d)
```

---

## Configuration Details

```json
{
  "model_type": "solar_open",
  "architectures": ["SolarOpenForCausalLM"],

  "hidden_size": 4096,
  "num_hidden_layers": 48,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,

  "intermediate_size": 10240,
  "moe_intermediate_size": 1280,
  "n_routed_experts": 128,
  "n_shared_experts": 1,
  "num_experts_per_tok": 8,

  "vocab_size": 196608,
  "max_position_embeddings": 131072,
  "rope_theta": 1000000,

  "rms_norm_eps": 1e-05,
  "torch_dtype": "bfloat16",
  "tie_word_embeddings": false,
  "norm_topk_prob": true,
  "routed_scaling_factor": 1.0
}
```

---

## Parameter Count Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parameter Distribution                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Embedding Layer:                                                │
│    vocab_size × hidden_size = 196,608 × 4,096                   │
│    = 805,306,368 params (~0.8B)                                 │
│                                                                  │
│  Per Transformer Layer:                                          │
│    Attention:                                                    │
│      Q: 4096 × 4096 = 16.7M                                     │
│      K: 4096 × 1024 = 4.2M (8 KV heads × 128)                   │
│      V: 4096 × 1024 = 4.2M                                      │
│      O: 4096 × 4096 = 16.7M                                     │
│      Subtotal: ~42M                                             │
│                                                                  │
│    MoE FFN:                                                      │
│      Shared Expert: 4096 × 10240 × 3 = 125.8M                   │
│      Routed Experts: 128 × (4096 × 1280 × 3) = 2.01B            │
│      Router: 4096 × 128 = 0.5M                                  │
│      Subtotal: ~2.14B                                           │
│                                                                  │
│    Layer Norms: ~16K                                             │
│    Per Layer Total: ~2.18B                                       │
│                                                                  │
│  Total (48 layers):                                              │
│    Embedding: 0.8B                                               │
│    Transformer: 48 × 2.18B = 104.6B                             │
│    Output Head: 0.8B                                             │
│    ─────────────────────────                                     │
│    Grand Total: ~102.6B parameters                               │
│                                                                  │
│  Active Parameters per Token:                                    │
│    Embedding + Attention + Shared + Top-8 Routed + Output       │
│    ≈ 12B parameters                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Depth Up-Scaling (DUS) - Solar's Innovation

Upstage의 이전 Solar 모델들(10.7B, 22B)은 **Depth Up-Scaling (DUS)** 기법을 사용했습니다:

```
┌─────────────────────────────────────────────────────────────────┐
│               Depth Up-Scaling (DUS) Method                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Base Model (32 Layers)          Copy of Base Model            │
│   ┌─────────────────────┐         ┌─────────────────────┐       │
│   │ Layer 1             │         │ Layer 1  ← Remove   │       │
│   │ Layer 2             │         │ Layer 2  ← Remove   │       │
│   │ ...                 │         │ ...      ← Remove   │       │
│   │ Layer 8  ← Remove   │         │ Layer 8  ← Remove   │       │
│   │ Layer 9  ← Remove   │         │ Layer 9             │       │
│   │ ...                 │         │ ...                 │       │
│   │ Layer 24            │         │ Layer 24            │       │
│   │ ...      ← Remove   │         │ ...                 │       │
│   │ Layer 32 ← Remove   │         │ Layer 32            │       │
│   └─────────────────────┘         └─────────────────────┘       │
│            │                                │                    │
│            │  Remove last 8 layers          │  Remove first 8   │
│            ▼                                ▼                    │
│   ┌─────────────────────┐         ┌─────────────────────┐       │
│   │ Layer 1-24          │         │ Layer 9-32          │       │
│   │ (24 layers)         │         │ (24 layers)         │       │
│   └─────────┬───────────┘         └──────────┬──────────┘       │
│             │                                │                   │
│             └────────────┬───────────────────┘                   │
│                          ▼                                       │
│                ┌─────────────────────┐                          │
│                │   Concatenate       │                          │
│                │   48 Layers Total   │                          │
│                └─────────────────────┘                          │
│                          ▼                                       │
│                ┌─────────────────────┐                          │
│                │ Continued Pretraining│                          │
│                │ on Additional Data  │                          │
│                └─────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Solar Open 100B는 DUS가 아닌 처음부터(from scratch) 학습되었습니다.**

---

## MoE Router Mechanism

```python
# Router Pseudocode
def route_tokens(hidden_states, router_weights):
    """
    hidden_states: [batch, seq, hidden_size]
    router_weights: [hidden_size, n_routed_experts]
    """
    # Compute routing scores
    router_logits = hidden_states @ router_weights  # [batch, seq, 128]

    # Apply softmax and select top-k
    routing_weights = softmax(router_logits, dim=-1)
    topk_weights, topk_indices = topk(routing_weights, k=8)

    # Normalize selected weights
    if norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # Shared expert always processes all tokens
    shared_output = shared_expert(hidden_states)

    # Routed experts process based on selection
    routed_output = sparse_moe(hidden_states, topk_weights, topk_indices)

    return shared_output + routed_output
```

---

## Memory & Hardware Requirements

### VRAM Estimation

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRAM Requirements                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model Weights (BF16):                                           │
│    102.6B × 2 bytes = 205.2 GB                                  │
│                                                                  │
│  KV Cache (128K context, batch=1, BF16):                        │
│    2 × 48 layers × 8 heads × 128 dim × 131072 seq × 2 bytes    │
│    = ~25.6 GB                                                   │
│                                                                  │
│  Activations & Overhead: ~20 GB                                  │
│                                                                  │
│  Total: ~250 GB VRAM                                             │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Recommended Hardware:                                           │
│    - Minimum: 4× NVIDIA A100 80GB                               │
│    - Optimal: 8× NVIDIA A100/H100 80GB                          │
│    - For vLLM: 8 GPUs recommended                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inference Configuration

### Recommended Generation Parameters

```python
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 4096,
    "repetition_penalty": 1.1,
}
```

### vLLM Deployment

```python
from vllm import LLM, SamplingParams

# 8 GPU setup
llm = LLM(
    model="upstage/Solar-Open-100B",
    tensor_parallel_size=8,
    max_model_len=131072,
    trust_remote_code=True,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    max_tokens=4096,
)

outputs = llm.generate(prompts, sampling_params)
```

---

## Comparison with Other Models

| Model | Total Params | Active Params | Architecture | Context |
|-------|--------------|---------------|--------------|---------|
| **Solar Open 100B** | 102.6B | 12B | MoE (129 experts) | 128K |
| Mixtral 8x22B | 141B | 39B | MoE (8 experts) | 65K |
| Llama 3.1 405B | 405B | 405B | Dense | 128K |
| DeepSeek-V3 | 671B | 37B | MoE (256+1 experts) | 128K |
| Qwen2.5-72B | 72B | 72B | Dense | 128K |

---

## Key Innovations

### 1. Efficient Expert Design
- **1 Shared + 128 Routed**: 공유 전문가가 공통 지식 처리
- **Top-8 Selection**: 토큰당 8개 전문가만 활성화

### 2. Extended Context
- **128K tokens**: RoPE theta 1,000,000으로 긴 컨텍스트 지원
- **GQA**: 8배 메모리 효율 향상

### 3. Training Efficiency
- **19.7T tokens**: 광범위한 지식 커버리지
- **B200 GPUs**: 최신 하드웨어로 효율적 학습

---

## References

- [Solar Open 100B - HuggingFace](https://huggingface.co/upstage/Solar-Open-100B)
- [SOLAR 10.7B Paper - arXiv](https://arxiv.org/abs/2312.15166)
- [Upstage AI](https://www.upstage.ai/)

---

## License

This analysis is provided under MIT License.
Solar Open 100B model is licensed under [Solar-Apache License 2.0](https://huggingface.co/upstage/Solar-Open-100B/blob/main/LICENSE).

---

*Last Updated: 2025-01-01*
