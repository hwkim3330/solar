"""
Solar Open 100B - Parameter Analysis

이 스크립트는 Solar Open 100B 모델의 파라미터 구조를 분석합니다.
실제 모델 로딩 없이 config 기반으로 파라미터 수를 계산합니다.
"""

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class SolarOpenConfig:
    """Solar Open 100B Configuration"""
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 10240          # Shared expert FFN
    moe_intermediate_size: int = 1280       # Routed expert FFN
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    vocab_size: int = 196608
    max_position_embeddings: int = 131072
    rope_theta: float = 1000000


def count_embedding_params(config: SolarOpenConfig) -> Dict[str, int]:
    """Embedding layer 파라미터 계산"""
    # Token embeddings
    embed_tokens = config.vocab_size * config.hidden_size

    return {
        "embed_tokens": embed_tokens,
        "total": embed_tokens
    }


def count_attention_params(config: SolarOpenConfig) -> Dict[str, int]:
    """Single attention layer 파라미터 계산"""
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    # Query projection: hidden_size -> n_heads * head_dim
    q_proj = hidden * (n_heads * head_dim)

    # Key projection: hidden_size -> n_kv_heads * head_dim (GQA)
    k_proj = hidden * (n_kv_heads * head_dim)

    # Value projection: hidden_size -> n_kv_heads * head_dim (GQA)
    v_proj = hidden * (n_kv_heads * head_dim)

    # Output projection: n_heads * head_dim -> hidden_size
    o_proj = (n_heads * head_dim) * hidden

    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "total": q_proj + k_proj + v_proj + o_proj
    }


def count_moe_params(config: SolarOpenConfig) -> Dict[str, int]:
    """MoE Feed-Forward layer 파라미터 계산"""
    hidden = config.hidden_size

    # Shared Expert: SwiGLU (gate + up + down)
    # gate_proj: hidden -> intermediate
    # up_proj: hidden -> intermediate
    # down_proj: intermediate -> hidden
    shared_gate = hidden * config.intermediate_size
    shared_up = hidden * config.intermediate_size
    shared_down = config.intermediate_size * hidden
    shared_total = (shared_gate + shared_up + shared_down) * config.n_shared_experts

    # Routed Experts: 각 expert는 더 작은 intermediate size 사용
    routed_gate = hidden * config.moe_intermediate_size
    routed_up = hidden * config.moe_intermediate_size
    routed_down = config.moe_intermediate_size * hidden
    routed_per_expert = routed_gate + routed_up + routed_down
    routed_total = routed_per_expert * config.n_routed_experts

    # Router: hidden -> n_routed_experts
    router = hidden * config.n_routed_experts

    return {
        "shared_experts": shared_total,
        "routed_experts": routed_total,
        "router": router,
        "per_routed_expert": routed_per_expert,
        "total": shared_total + routed_total + router
    }


def count_layer_norm_params(config: SolarOpenConfig) -> Dict[str, int]:
    """Layer normalization 파라미터 계산"""
    # RMSNorm: only scale parameter (no bias)
    input_layernorm = config.hidden_size
    post_attention_layernorm = config.hidden_size

    return {
        "input_layernorm": input_layernorm,
        "post_attention_layernorm": post_attention_layernorm,
        "total": input_layernorm + post_attention_layernorm
    }


def count_output_params(config: SolarOpenConfig) -> Dict[str, int]:
    """Output layer 파라미터 계산"""
    # Final layer norm
    final_norm = config.hidden_size

    # LM head (unembedding) - 보통 embedding과 공유하지 않음
    lm_head = config.hidden_size * config.vocab_size

    return {
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": final_norm + lm_head
    }


def analyze_model(config: SolarOpenConfig) -> Dict:
    """전체 모델 분석"""

    # 각 컴포넌트 파라미터 계산
    embedding = count_embedding_params(config)
    attention = count_attention_params(config)
    moe = count_moe_params(config)
    layer_norm = count_layer_norm_params(config)
    output = count_output_params(config)

    # Per-layer 총합
    per_layer = attention["total"] + moe["total"] + layer_norm["total"]

    # 전체 모델 총합
    total_params = (
        embedding["total"] +
        per_layer * config.num_hidden_layers +
        output["total"]
    )

    # Active parameters per token
    # Shared expert + Top-8 routed experts + attention + embedding/output
    active_moe = (
        (config.hidden_size * config.intermediate_size * 3) * config.n_shared_experts +
        (config.hidden_size * config.moe_intermediate_size * 3) * config.num_experts_per_tok
    )
    active_per_layer = attention["total"] + active_moe + layer_norm["total"]
    active_params = (
        embedding["total"] +
        active_per_layer * config.num_hidden_layers +
        output["total"]
    )

    return {
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "n_routed_experts": config.n_routed_experts,
            "n_shared_experts": config.n_shared_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "vocab_size": config.vocab_size,
            "max_seq_length": config.max_position_embeddings,
        },
        "parameters": {
            "embedding": embedding,
            "attention_per_layer": attention,
            "moe_per_layer": moe,
            "layer_norm_per_layer": layer_norm,
            "output": output,
            "per_layer_total": per_layer,
            "total": total_params,
            "active_per_token": active_params,
        },
        "summary": {
            "total_params_billions": total_params / 1e9,
            "active_params_billions": active_params / 1e9,
            "efficiency_ratio": active_params / total_params,
        }
    }


def format_number(n: int) -> str:
    """큰 숫자를 읽기 쉽게 포맷"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def print_analysis(analysis: Dict):
    """분석 결과 출력"""
    config = analysis["config"]
    params = analysis["parameters"]
    summary = analysis["summary"]

    print("=" * 70)
    print("                    Solar Open 100B - Parameter Analysis")
    print("=" * 70)

    print("\n[Model Configuration]")
    print(f"  Hidden Size:          {config['hidden_size']}")
    print(f"  Number of Layers:     {config['num_layers']}")
    print(f"  Attention Heads:      {config['num_attention_heads']}")
    print(f"  KV Heads (GQA):       {config['num_kv_heads']}")
    print(f"  Routed Experts:       {config['n_routed_experts']}")
    print(f"  Shared Experts:       {config['n_shared_experts']}")
    print(f"  Experts per Token:    {config['num_experts_per_tok']}")
    print(f"  Vocabulary Size:      {config['vocab_size']:,}")
    print(f"  Max Sequence Length:  {config['max_seq_length']:,}")

    print("\n[Parameter Breakdown]")
    print(f"  Embedding Layer:      {format_number(params['embedding']['total'])}")
    print(f"  Per Attention Layer:  {format_number(params['attention_per_layer']['total'])}")
    print(f"    - Q projection:     {format_number(params['attention_per_layer']['q_proj'])}")
    print(f"    - K projection:     {format_number(params['attention_per_layer']['k_proj'])}")
    print(f"    - V projection:     {format_number(params['attention_per_layer']['v_proj'])}")
    print(f"    - O projection:     {format_number(params['attention_per_layer']['o_proj'])}")

    print(f"  Per MoE Layer:        {format_number(params['moe_per_layer']['total'])}")
    print(f"    - Shared Experts:   {format_number(params['moe_per_layer']['shared_experts'])}")
    print(f"    - Routed Experts:   {format_number(params['moe_per_layer']['routed_experts'])}")
    print(f"    - Router:           {format_number(params['moe_per_layer']['router'])}")
    print(f"    - Per Expert:       {format_number(params['moe_per_layer']['per_routed_expert'])}")

    print(f"  Per Layer Norm:       {format_number(params['layer_norm_per_layer']['total'])}")
    print(f"  Output Layer:         {format_number(params['output']['total'])}")
    print(f"    - Final Norm:       {format_number(params['output']['final_norm'])}")
    print(f"    - LM Head:          {format_number(params['output']['lm_head'])}")

    print("\n[Summary]")
    print(f"  Per Layer Total:      {format_number(params['per_layer_total'])}")
    print(f"  Total Parameters:     {summary['total_params_billions']:.2f}B")
    print(f"  Active per Token:     {summary['active_params_billions']:.2f}B")
    print(f"  Efficiency Ratio:     {summary['efficiency_ratio']*100:.1f}%")

    print("\n[Memory Estimation (BF16)]")
    total_bytes = params['total'] * 2  # 2 bytes per BF16
    print(f"  Model Weights:        {total_bytes / 1e9:.1f} GB")
    print(f"  Recommended VRAM:     {total_bytes / 1e9 * 1.2:.1f} GB (with overhead)")

    print("=" * 70)


def main():
    # Load config
    config = SolarOpenConfig()

    # Analyze
    analysis = analyze_model(config)

    # Print results
    print_analysis(analysis)

    # Save to JSON
    with open("analysis_result.json", "w") as f:
        # Convert to serializable format
        result = {
            "config": analysis["config"],
            "parameters": {
                k: {kk: int(vv) if isinstance(vv, (int, float)) else vv
                    for kk, vv in v.items()}
                for k, v in analysis["parameters"].items()
            },
            "summary": analysis["summary"]
        }
        json.dump(result, f, indent=2)
        print("\nResults saved to analysis_result.json")


if __name__ == "__main__":
    main()
