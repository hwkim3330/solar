"""
Solar Open 100B - VRAM Calculator

다양한 배치 사이즈와 시퀀스 길이에서 필요한 VRAM을 계산합니다.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Solar Open 100B Configuration"""
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1280
    n_routed_experts: int = 128
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    vocab_size: int = 196608
    max_position_embeddings: int = 131072
    total_params: float = 102.6e9
    active_params: float = 12e9


def calculate_model_weights_memory(
    config: ModelConfig,
    dtype_bytes: int = 2  # BF16 = 2 bytes
) -> float:
    """모델 가중치 메모리 계산 (GB)"""
    return config.total_params * dtype_bytes / 1e9


def calculate_kv_cache_memory(
    config: ModelConfig,
    batch_size: int,
    seq_length: int,
    dtype_bytes: int = 2
) -> float:
    """KV Cache 메모리 계산 (GB)

    KV Cache = 2 * num_layers * num_kv_heads * head_dim * seq_length * batch_size * dtype_bytes
    """
    kv_cache = (
        2 *  # K and V
        config.num_hidden_layers *
        config.num_kv_heads *
        config.head_dim *
        seq_length *
        batch_size *
        dtype_bytes
    )
    return kv_cache / 1e9


def calculate_activation_memory(
    config: ModelConfig,
    batch_size: int,
    seq_length: int,
    dtype_bytes: int = 2
) -> float:
    """Activation 메모리 추정 (GB)

    주요 activation:
    - Attention: batch * seq * hidden
    - MoE routing: batch * seq * n_experts
    - FFN activations: batch * seq * intermediate
    """
    # Attention activations
    attention_act = batch_size * seq_length * config.hidden_size * config.num_hidden_layers

    # MoE routing scores
    routing_act = batch_size * seq_length * config.n_routed_experts * config.num_hidden_layers

    # FFN intermediate (only active experts)
    ffn_act = batch_size * seq_length * (
        config.intermediate_size +  # Shared expert
        config.moe_intermediate_size * config.num_experts_per_tok  # Active routed experts
    ) * config.num_hidden_layers

    total_activations = (attention_act + routing_act + ffn_act) * dtype_bytes
    return total_activations / 1e9


def calculate_total_memory(
    config: ModelConfig,
    batch_size: int,
    seq_length: int,
    dtype_bytes: int = 2,
    overhead_factor: float = 1.15  # 15% overhead for fragmentation, etc.
) -> Tuple[float, dict]:
    """총 VRAM 요구량 계산 (GB)"""

    weights = calculate_model_weights_memory(config, dtype_bytes)
    kv_cache = calculate_kv_cache_memory(config, batch_size, seq_length, dtype_bytes)
    activations = calculate_activation_memory(config, batch_size, seq_length, dtype_bytes)

    subtotal = weights + kv_cache + activations
    total = subtotal * overhead_factor

    breakdown = {
        "model_weights": weights,
        "kv_cache": kv_cache,
        "activations": activations,
        "subtotal": subtotal,
        "overhead": subtotal * (overhead_factor - 1),
        "total": total
    }

    return total, breakdown


def recommend_gpu_config(total_memory: float) -> str:
    """필요한 GPU 구성 추천"""
    gpu_configs = [
        (80, 1, "1x A100 80GB"),
        (80, 2, "2x A100 80GB"),
        (80, 4, "4x A100 80GB"),
        (80, 8, "8x A100 80GB"),
        (80, 16, "16x A100 80GB (2 nodes)"),
        (141, 8, "8x H100 141GB (NVL)"),
    ]

    for gpu_mem, num_gpus, name in gpu_configs:
        if total_memory <= gpu_mem * num_gpus * 0.9:  # 90% utilization
            return name

    return "Requires more than 16x A100 80GB or distributed setup"


def print_memory_analysis(config: ModelConfig):
    """메모리 분석 결과 출력"""
    print("=" * 70)
    print("              Solar Open 100B - VRAM Requirements Analysis")
    print("=" * 70)

    # 다양한 설정에서 테스트
    test_configs = [
        (1, 1024, "Short context (1K)"),
        (1, 4096, "Medium context (4K)"),
        (1, 32768, "Long context (32K)"),
        (1, 131072, "Full context (128K)"),
        (4, 4096, "Batch=4, 4K context"),
        (8, 2048, "Batch=8, 2K context"),
    ]

    print("\n[Model Weights]")
    weights_bf16 = calculate_model_weights_memory(config, dtype_bytes=2)
    weights_fp8 = calculate_model_weights_memory(config, dtype_bytes=1)
    print(f"  BF16 (2 bytes):       {weights_bf16:.1f} GB")
    print(f"  FP8 (1 byte):         {weights_fp8:.1f} GB")

    print("\n[Memory by Configuration]")
    print("-" * 70)
    print(f"{'Config':<25} {'Weights':>10} {'KV Cache':>10} {'Act':>10} {'Total':>10}")
    print("-" * 70)

    for batch, seq, name in test_configs:
        total, breakdown = calculate_total_memory(config, batch, seq)
        print(f"{name:<25} {breakdown['model_weights']:>9.1f}G {breakdown['kv_cache']:>9.1f}G "
              f"{breakdown['activations']:>9.1f}G {breakdown['total']:>9.1f}G")

    print("-" * 70)

    print("\n[GPU Recommendations]")
    print("-" * 70)

    for batch, seq, name in test_configs:
        total, _ = calculate_total_memory(config, batch, seq)
        gpu_rec = recommend_gpu_config(total)
        print(f"  {name:<25} -> {gpu_rec}")

    print("\n[Tensor Parallelism Strategies]")
    print("-" * 70)
    tp_configs = [2, 4, 8]
    for tp in tp_configs:
        per_gpu = weights_bf16 / tp
        print(f"  TP={tp}: {per_gpu:.1f} GB per GPU (weights only)")

    print("\n[Notes]")
    print("  - KV Cache scales linearly with sequence length and batch size")
    print("  - GQA reduces KV cache by 8x compared to MHA")
    print("  - MoE activations are sparse (only top-8 experts active)")
    print("  - Consider using Flash Attention for memory efficiency")
    print("  - vLLM recommended for production serving")
    print("=" * 70)


def main():
    config = ModelConfig()
    print_memory_analysis(config)


if __name__ == "__main__":
    main()
