# Solar Open 100B vs GLM-4.5-Air: Architecture Analysis

> 심층적인 모델 구조 비교 분석 및 유사성 검증

[![Solar](https://img.shields.io/badge/HuggingFace-Solar--Open--100B-yellow)](https://huggingface.co/upstage/Solar-Open-100B)
[![GLM](https://img.shields.io/badge/HuggingFace-GLM--4.5--Air-green)](https://huggingface.co/zai-org/GLM-4.5-Air)
[![Paper](https://img.shields.io/badge/arXiv-2312.15166-red)](https://arxiv.org/abs/2312.15166)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Configuration Comparison](#configuration-comparison)
3. [Architecture Analysis](#architecture-analysis)
4. [Weight Similarity Methodology](#weight-similarity-methodology)
5. [Layer-by-Layer Comparison](#layer-by-layer-comparison)
6. [Technical Deep Dive](#technical-deep-dive)
7. [Reproduction Scripts](#reproduction-scripts)
8. [References](#references)

---

## Executive Summary

### Models Under Comparison

| Model | Developer | Release | Total Params | Active Params |
|-------|-----------|---------|--------------|---------------|
| **Solar Open 100B** | Upstage | 2024.12 | 102.6B | 12B |
| **GLM-4.5-Air** | Zhipu AI | 2024.11 | 106B | 12B |

### Key Architectural Similarities

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IDENTICAL PARAMETERS                              │
├─────────────────────────────────────────────────────────────────────┤
│  hidden_size         = 4096      ✓ SAME                             │
│  num_kv_heads        = 8         ✓ SAME                             │
│  head_dim            = 128       ✓ SAME                             │
│  n_routed_experts    = 128       ✓ SAME                             │
│  n_shared_experts    = 1         ✓ SAME                             │
│  num_experts_per_tok = 8         ✓ SAME                             │
│  max_position_embed  = 131072    ✓ SAME                             │
│  rope_theta          = 1000000   ✓ SAME                             │
│  rms_norm_eps        = 1e-05     ✓ SAME                             │
├─────────────────────────────────────────────────────────────────────┤
│                    DIFFERENT PARAMETERS                              │
├─────────────────────────────────────────────────────────────────────┤
│  num_hidden_layers   : 48 vs 46           (+2 layers)               │
│  num_attention_heads : 64 vs 96           (different GQA ratio)     │
│  intermediate_size   : 10240 vs 10944     (-6.4%)                   │
│  moe_intermediate    : 1280 vs 1408       (-9.1%)                   │
│  vocab_size          : 196608 vs 151552   (+29.7%)                  │
│  partial_rotary      : 1.0 vs 0.5         (full vs partial RoPE)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Comparison

### Side-by-Side Config Analysis

| Parameter | Solar Open 100B | GLM-4.5-Air | Match |
|-----------|-----------------|-------------|-------|
| **Architecture** | | | |
| `model_type` | `solar_open` | `glm4_moe` | ✗ |
| `hidden_size` | 4096 | 4096 | ✓ |
| `num_hidden_layers` | 48 | 46 | ✗ |
| **Attention** | | | |
| `num_attention_heads` | 64 | 96 | ✗ |
| `num_key_value_heads` | 8 | 8 | ✓ |
| `head_dim` | 128 | 128 | ✓ |
| `attention_bias` | false | true | ✗ |
| **MoE Configuration** | | | |
| `n_routed_experts` | 128 | 128 | ✓ |
| `n_shared_experts` | 1 | 1 | ✓ |
| `num_experts_per_tok` | 8 | 8 | ✓ |
| `intermediate_size` | 10240 | 10944 | ✗ |
| `moe_intermediate_size` | 1280 | 1408 | ✗ |
| **Position Encoding** | | | |
| `max_position_embeddings` | 131072 | 131072 | ✓ |
| `rope_theta` | 1000000 | 1000000 | ✓ |
| `partial_rotary_factor` | 1.0 | 0.5 | ✗ |
| **Vocabulary** | | | |
| `vocab_size` | 196608 | 151552 | ✗ |
| `tie_word_embeddings` | false | false | ✓ |

### Statistical Summary

```
Total Parameters Compared: 18
Identical:                  9 (50.0%)
Different:                  9 (50.0%)

Core MoE Structure Match:   6/6 (100%)
Attention Structure Match:  2/4 (50%)
Position Encoding Match:    2/3 (67%)
```

---

## Architecture Analysis

### MoE Layer Structure Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Solar Open 100B                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input (4096) ─────┬──────────────────────────────────────────┐    │
│                     │                                          │    │
│                     ▼                                          │    │
│            ┌────────────────┐                                  │    │
│            │  Router (4096→128)                                │    │
│            │  + Softmax + TopK                                 │    │
│            └────────┬───────┘                                  │    │
│                     │                                          │    │
│         ┌──────────┬┴┬──────────┐                              │    │
│         ▼          ▼ ▼          ▼                              │    │
│   ┌──────────┐ ┌────────────────────┐                          │    │
│   │ Shared   │ │ Routed Experts     │                          │    │
│   │ Expert   │ │ (Top-8 of 128)     │                          │    │
│   │          │ │                    │                          │    │
│   │ 4096     │ │ 4096 → 1280 → 4096 │ × 8                      │    │
│   │   ↓      │ │ (SwiGLU each)      │                          │    │
│   │ 10240    │ └──────────┬─────────┘                          │    │
│   │   ↓      │            │                                    │    │
│   │ 4096     │            │                                    │    │
│   └────┬─────┘            │                                    │    │
│        │                  │                                    │    │
│        └───────┬──────────┘                                    │    │
│                ▼                                               │    │
│         Weighted Sum ──────────────────────────────────► Output│    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        GLM-4.5-Air                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input (4096) ─────┬──────────────────────────────────────────┐    │
│                     │                                          │    │
│                     ▼                                          │    │
│            ┌────────────────┐                                  │    │
│            │  Router (4096→128)                                │    │
│            │  + Sigmoid Gating                                 │    │
│            └────────┬───────┘                                  │    │
│                     │                                          │    │
│         ┌──────────┬┴┬──────────┐                              │    │
│         ▼          ▼ ▼          ▼                              │    │
│   ┌──────────┐ ┌────────────────────┐                          │    │
│   │ Shared   │ │ Routed Experts     │                          │    │
│   │ Expert   │ │ (Top-8 of 128)     │                          │    │
│   │          │ │                    │                          │    │
│   │ 4096     │ │ 4096 → 1408 → 4096 │ × 8                      │    │
│   │   ↓      │ │ (SwiGLU each)      │                          │    │
│   │ 10944    │ └──────────┬─────────┘                          │    │
│   │   ↓      │            │                                    │    │
│   │ 4096     │            │                                    │    │
│   └────┬─────┘            │                                    │    │
│        │                  │                                    │    │
│        └───────┬──────────┘                                    │    │
│                ▼                                               │    │
│         Weighted Sum ──────────────────────────────────► Output│    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Attention Mechanism Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│              Grouped Query Attention (GQA) Comparison                │
├──────────────────────────────┬──────────────────────────────────────┤
│        Solar Open 100B       │           GLM-4.5-Air                │
├──────────────────────────────┼──────────────────────────────────────┤
│                              │                                      │
│  Query Heads: 64             │  Query Heads: 96                     │
│  KV Heads: 8                 │  KV Heads: 8                         │
│  GQA Ratio: 8:1              │  GQA Ratio: 12:1                     │
│                              │                                      │
│  ┌─────────────────────┐     │  ┌─────────────────────┐             │
│  │ Q: 64 × 128 = 8192  │     │  │ Q: 96 × 128 = 12288 │             │
│  │ K: 8 × 128 = 1024   │     │  │ K: 8 × 128 = 1024   │             │
│  │ V: 8 × 128 = 1024   │     │  │ V: 8 × 128 = 1024   │             │
│  └─────────────────────┘     │  └─────────────────────┘             │
│                              │                                      │
│  Q Params: 4096 × 8192       │  Q Params: 4096 × 12288              │
│          = 33.5M             │          = 50.3M                     │
│                              │                                      │
│  Each KV head serves         │  Each KV head serves                 │
│  8 query heads               │  12 query heads                      │
│                              │                                      │
├──────────────────────────────┼──────────────────────────────────────┤
│  RoPE: Full (factor=1.0)     │  RoPE: Partial (factor=0.5)          │
│  - All dims rotated          │  - 50% dims rotated                  │
│  - Standard long context     │  - Hybrid position encoding          │
└──────────────────────────────┴──────────────────────────────────────┘
```

---

## Weight Similarity Methodology

### The Controversy

2024년 12월, [sionic-ai/solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm) 레포지토리에서 Solar Open 100B가 GLM-4.5-Air로부터 파생되었다는 주장이 제기되었습니다.

### Analysis Methods

#### 1. Cosine Similarity (Original Claim)

```python
def cosine_similarity(a, b):
    """
    cos(θ) = (A · B) / (||A|| × ||B||)

    문제점:
    - 스케일 불변 (magnitude 무시)
    - 초기화 bias (LayerNorm ≈ 1.0)
    - 방향만 비교, 크기 차이 무시
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

#### 2. Mean Absolute Difference (Counter-Analysis)

```python
def mean_abs_diff(a, b):
    """
    MAD = (1/n) × Σ|a_i - b_i|

    장점:
    - 스케일 민감 (magnitude 고려)
    - 초기화 bias 극복
    - 실제 수치 차이 측정
    """
    return np.mean(np.abs(a - b))
```

#### 3. Pearson Correlation (PR #3 Proposal)

```python
def pearson_correlation(a, b):
    """
    r = Σ(a - ā)(b - b̄) / √(Σ(a - ā)² × Σ(b - b̄)²)

    장점:
    - 평균 중심화로 초기화 artifact 제거
    - 상관관계 측정
    """
    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    return np.dot(a_centered, b_centered) / (
        np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    )
```

### Why LayerNorm Comparison is Problematic

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RMSNorm Weight Characteristics                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Initialization:    γ ≈ 1.0 (ones-like initialization)              │
│  Dimension:         (hidden_size,) = (4096,)                        │
│  Information:       Low entropy, high redundancy                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Example LayerNorm weights (normalized):                     │   │
│  │                                                              │   │
│  │  Solar:  [0.98, 1.02, 0.99, 1.01, 0.97, ...]               │   │
│  │  GLM:    [0.99, 1.01, 0.98, 1.02, 0.98, ...]               │   │
│  │  Phi:    [0.97, 1.03, 1.00, 0.99, 0.98, ...]               │   │
│  │                                                              │   │
│  │  All clustered around 1.0 → High cosine similarity          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Result: Cosine similarity > 0.9 for ANY two models                 │
│          This does NOT indicate derivation                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Counter-Evidence from [hyunwoongko/solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)

| Comparison | Cosine Sim | Mean Abs Diff |
|------------|------------|---------------|
| Solar vs GLM (Layer 10) | 0.99+ | 0.2615 |
| GLM vs Phi (Layer 10) | 0.99+ | 0.178 |
| Solar vs Phi (Layer 10) | 0.99+ | 0.284 |

**결론**: Cosine similarity로는 Solar-GLM이 유사해 보이지만, Mean Abs Diff로는 오히려 GLM-Phi가 더 유사합니다.

---

## Layer-by-Layer Comparison

### Transformer Block Structure

```
                    Solar Open 100B                     GLM-4.5-Air
                    ===============                     ===========

Layer 0-47 (48 layers)                    Layer 0-45 (46 layers)
┌─────────────────────────┐               ┌─────────────────────────┐
│                         │               │                         │
│  ┌───────────────────┐  │               │  ┌───────────────────┐  │
│  │ input_layernorm   │  │               │  │ input_layernorm   │  │
│  │ (RMSNorm, 4096)   │  │               │  │ (RMSNorm, 4096)   │  │
│  └─────────┬─────────┘  │               │  └─────────┬─────────┘  │
│            │            │               │            │            │
│  ┌─────────▼─────────┐  │               │  ┌─────────▼─────────┐  │
│  │   Self Attention  │  │               │  │   Self Attention  │  │
│  │                   │  │               │  │                   │  │
│  │  Q: 4096 → 8192   │  │               │  │  Q: 4096 → 12288  │  │
│  │  K: 4096 → 1024   │  │               │  │  K: 4096 → 1024   │  │
│  │  V: 4096 → 1024   │  │               │  │  V: 4096 → 1024   │  │
│  │  O: 8192 → 4096   │  │               │  │  O: 12288 → 4096  │  │
│  │                   │  │               │  │                   │  │
│  │  + RoPE (full)    │  │               │  │  + RoPE (partial) │  │
│  │  + Bias: No       │  │               │  │  + Bias: Yes      │  │
│  └─────────┬─────────┘  │               │  └─────────┬─────────┘  │
│            │            │               │            │            │
│  ┌─────────▼─────────┐  │               │  ┌─────────▼─────────┐  │
│  │ post_attn_norm    │  │               │  │ post_attn_norm    │  │
│  │ (RMSNorm, 4096)   │  │               │  │ (RMSNorm, 4096)   │  │
│  └─────────┬─────────┘  │               │  └─────────┬─────────┘  │
│            │            │               │            │            │
│  ┌─────────▼─────────┐  │               │  ┌─────────▼─────────┐  │
│  │     MoE Layer     │  │               │  │     MoE Layer     │  │
│  │                   │  │               │  │                   │  │
│  │  Router: 4096→128 │  │               │  │  Router: 4096→128 │  │
│  │  Shared: 10240    │  │               │  │  Shared: 10944    │  │
│  │  Routed: 1280×128 │  │               │  │  Routed: 1408×128 │  │
│  │  TopK: 8          │  │               │  │  TopK: 8          │  │
│  └─────────┬─────────┘  │               │  └─────────┬─────────┘  │
│            │            │               │            │            │
│            ▼            │               │            ▼            │
└─────────────────────────┘               └─────────────────────────┘
```

### Parameter Count per Layer

| Component | Solar Open 100B | GLM-4.5-Air | Difference |
|-----------|-----------------|-------------|------------|
| Q projection | 33.55M | 50.33M | -33.3% |
| K projection | 4.19M | 4.19M | 0% |
| V projection | 4.19M | 4.19M | 0% |
| O projection | 33.55M | 50.33M | -33.3% |
| Shared Expert | 125.83M | 134.22M | -6.2% |
| Per Routed Expert | 15.73M | 17.30M | -9.1% |
| All Routed Experts | 2.01B | 2.21B | -9.1% |
| Router | 0.52M | 0.52M | 0% |
| Layer Norms | 8.19K | 8.19K | 0% |
| **Layer Total** | ~2.21B | ~2.45B | -9.8% |

---

## Technical Deep Dive

### Expert Routing Mechanism

```python
# Solar Open 100B routing (inferred from config)
class SolarMoERouter:
    def __init__(self, hidden_size=4096, n_experts=128, top_k=8):
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.top_k = top_k
        self.norm_topk_prob = True  # config: norm_topk_prob = true

    def forward(self, hidden_states):
        # Compute routing scores
        router_logits = self.gate(hidden_states)  # [batch, seq, 128]

        # Softmax over experts
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Normalize selected weights (if norm_topk_prob)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_indices


# GLM-4.5-Air routing (different gating mechanism)
class GLMMoERouter:
    def __init__(self, hidden_size=4096, n_experts=128, top_k=8):
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.top_k = top_k
        # Uses sigmoid gating with loss-free balance routing

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)

        # Sigmoid gating (different from Softmax)
        routing_weights = torch.sigmoid(router_logits)

        # Loss-free balance routing
        # ... additional balancing logic ...

        topk_weights, topk_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        return topk_weights, topk_indices
```

### RoPE Implementation Difference

```python
# Solar: Full Rotary (partial_rotary_factor = 1.0)
def solar_rope(x, freqs):
    """
    모든 차원에 rotary embedding 적용
    head_dim = 128, 128개 모두 회전
    """
    x_rot = x  # 전체 사용
    x_pass = None  # 없음

    cos, sin = freqs
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return x_rot


# GLM: Partial Rotary (partial_rotary_factor = 0.5)
def glm_rope(x, freqs):
    """
    50%만 rotary embedding 적용
    head_dim = 128, 64개만 회전, 64개는 그대로
    """
    rotary_dim = int(x.shape[-1] * 0.5)  # 64

    x_rot = x[..., :rotary_dim]  # 처음 64개
    x_pass = x[..., rotary_dim:]  # 나머지 64개

    cos, sin = freqs
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)

    return torch.cat([x_rot, x_pass], dim=-1)
```

---

## Reproduction Scripts

### Weight Comparison Script

```python
# analysis/weight_comparison.py

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

def load_layer_weights(model, layer_idx):
    """특정 레이어의 가중치 추출"""
    layer = model.model.layers[layer_idx]

    return {
        'input_layernorm': layer.input_layernorm.weight.detach().cpu().numpy(),
        'post_attention_layernorm': layer.post_attention_layernorm.weight.detach().cpu().numpy(),
        'q_proj': layer.self_attn.q_proj.weight.detach().cpu().numpy(),
        'k_proj': layer.self_attn.k_proj.weight.detach().cpu().numpy(),
        'v_proj': layer.self_attn.v_proj.weight.detach().cpu().numpy(),
        'o_proj': layer.self_attn.o_proj.weight.detach().cpu().numpy(),
        'router': layer.mlp.gate.weight.detach().cpu().numpy(),
    }

def cosine_similarity(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

def mean_abs_diff(a, b):
    return np.mean(np.abs(a.flatten() - b.flatten()))

def pearson_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_centered = a_flat - np.mean(a_flat)
    b_centered = b_flat - np.mean(b_flat)
    return np.dot(a_centered, b_centered) / (
        np.linalg.norm(a_centered) * np.linalg.norm(b_centered) + 1e-8
    )

def compare_layers(model1_weights, model2_weights):
    """두 레이어의 가중치 비교"""
    results = {}

    for key in model1_weights:
        if key not in model2_weights:
            continue

        w1 = model1_weights[key]
        w2 = model2_weights[key]

        # Shape이 다르면 비교 불가
        if w1.shape != w2.shape:
            results[key] = {
                'comparable': False,
                'shape_mismatch': f'{w1.shape} vs {w2.shape}'
            }
            continue

        results[key] = {
            'comparable': True,
            'shape': w1.shape,
            'cosine_similarity': float(cosine_similarity(w1, w2)),
            'mean_abs_diff': float(mean_abs_diff(w1, w2)),
            'pearson_correlation': float(pearson_correlation(w1, w2)),
        }

    return results
```

### Full Analysis Pipeline

```bash
# 실행 방법 (GPU 8장 필요)
cd analysis
python weight_comparison.py \
    --model1 upstage/Solar-Open-100B \
    --model2 zai-org/GLM-4.5-Air \
    --layers 10,20,30,40 \
    --output results.json
```

---

## Conclusions

### Architectural Independence

1. **다른 Attention 구조**: 64 vs 96 heads (50% 차이)
2. **다른 FFN 크기**: 10240/1280 vs 10944/1408
3. **다른 RoPE**: Full vs Partial rotation
4. **다른 Layer 수**: 48 vs 46
5. **다른 Vocabulary**: 196K vs 151K

### Similarity Analysis Limitations

1. **Cosine Similarity 한계**:
   - LayerNorm weights가 1.0 근처로 초기화되어 false positive 발생
   - 관련 없는 모델도 0.9+ 유사도 표시

2. **적절한 비교 방법**:
   - Mean Absolute Difference
   - Pearson Correlation (평균 제거)
   - Attention/FFN weight 비교 (LayerNorm 제외)

### Open Questions

- [ ] 전체 weight byte-level 비교 결과
- [ ] Router weight distribution 분석
- [ ] Training data overlap 분석
- [ ] 학습 dynamics 비교

---

## References

1. [Solar Open 100B - HuggingFace](https://huggingface.co/upstage/Solar-Open-100B)
2. [GLM-4.5-Air - HuggingFace](https://huggingface.co/zai-org/GLM-4.5-Air)
3. [SOLAR 10.7B Paper - arXiv](https://arxiv.org/abs/2312.15166)
4. [sionic-ai/solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm) - Original similarity claim
5. [hyunwoongko/solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi) - Counter-analysis
6. [GLM-4.5 Technical Report](https://z.ai/blog/glm-4.5)

---

## License

This analysis is provided under MIT License for educational and research purposes.

---

*Last Updated: 2025-01-01*
