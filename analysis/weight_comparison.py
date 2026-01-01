"""
Solar Open 100B vs GLM-4.5-Air - Weight Comparison Analysis

다양한 유사도 메트릭을 사용하여 두 모델의 가중치를 비교합니다.
실제 모델 로딩에는 8x A100 GPU가 필요합니다.

Usage:
    python weight_comparison.py --model1 upstage/Solar-Open-100B \
                                --model2 zai-org/GLM-4.5-Air \
                                --layers 10,20,30,40 \
                                --output results.json
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings


@dataclass
class ComparisonResult:
    """단일 가중치 비교 결과"""
    weight_name: str
    shape1: Tuple[int, ...]
    shape2: Tuple[int, ...]
    comparable: bool
    cosine_similarity: Optional[float] = None
    mean_abs_diff: Optional[float] = None
    pearson_correlation: Optional[float] = None
    max_abs_diff: Optional[float] = None
    l2_distance: Optional[float] = None
    shape_mismatch: Optional[str] = None


@dataclass
class LayerComparisonResult:
    """레이어 전체 비교 결과"""
    layer_idx: int
    weights: Dict[str, ComparisonResult]
    summary: Dict[str, float]


class SimilarityMetrics:
    """유사도 측정 메트릭 모음"""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        코사인 유사도: cos(θ) = (A · B) / (||A|| × ||B||)

        특징:
        - 범위: [-1, 1]
        - 스케일 불변 (magnitude 무시)
        - 방향만 비교

        한계:
        - LayerNorm weights (≈1.0 초기화)에서 false positive 발생
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)

        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))

    @staticmethod
    def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        """
        평균 절대 차이: MAD = (1/n) × Σ|a_i - b_i|

        특징:
        - 범위: [0, ∞)
        - 스케일 민감
        - 실제 수치 차이 측정

        장점:
        - 초기화 bias 극복
        - 직관적 해석
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        return float(np.mean(np.abs(a_flat - b_flat)))

    @staticmethod
    def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """
        피어슨 상관계수: r = Σ(a - ā)(b - b̄) / √(Σ(a - ā)² × Σ(b - b̄)²)

        특징:
        - 범위: [-1, 1]
        - 평균 중심화로 초기화 artifact 제거

        장점:
        - sionic-ai PR #3에서 제안된 방법
        - LayerNorm 비교에 더 적합
        """
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)

        a_centered = a_flat - np.mean(a_flat)
        b_centered = b_flat - np.mean(b_flat)

        norm_a = np.linalg.norm(a_centered)
        norm_b = np.linalg.norm(b_centered)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(np.dot(a_centered, b_centered) / (norm_a * norm_b))

    @staticmethod
    def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
        """최대 절대 차이"""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        return float(np.max(np.abs(a_flat - b_flat)))

    @staticmethod
    def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
        """L2 (유클리드) 거리"""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        return float(np.linalg.norm(a_flat - b_flat))


def compare_weights(w1: np.ndarray, w2: np.ndarray, name: str) -> ComparisonResult:
    """두 가중치 텐서 비교"""

    shape1 = tuple(w1.shape)
    shape2 = tuple(w2.shape)

    # Shape이 다르면 비교 불가
    if shape1 != shape2:
        return ComparisonResult(
            weight_name=name,
            shape1=shape1,
            shape2=shape2,
            comparable=False,
            shape_mismatch=f"{shape1} vs {shape2}"
        )

    metrics = SimilarityMetrics()

    return ComparisonResult(
        weight_name=name,
        shape1=shape1,
        shape2=shape2,
        comparable=True,
        cosine_similarity=metrics.cosine_similarity(w1, w2),
        mean_abs_diff=metrics.mean_abs_diff(w1, w2),
        pearson_correlation=metrics.pearson_correlation(w1, w2),
        max_abs_diff=metrics.max_abs_diff(w1, w2),
        l2_distance=metrics.l2_distance(w1, w2),
    )


def simulate_layernorm_weights(size: int = 4096, seed: int = None) -> np.ndarray:
    """
    RMSNorm 가중치 시뮬레이션

    실제 학습된 LayerNorm weights는 1.0 근처에 분포
    """
    if seed is not None:
        np.random.seed(seed)

    # 1.0 근처의 작은 변동
    weights = 1.0 + np.random.normal(0, 0.05, size)
    return weights.astype(np.float32)


def simulate_comparison_demo():
    """
    시뮬레이션을 통한 Cosine Similarity vs MAD 비교 데모

    LayerNorm weights가 왜 Cosine Similarity에서 높은 값을 보이는지 시연
    """
    print("=" * 70)
    print("           LayerNorm Weight Similarity Demo")
    print("=" * 70)
    print()

    # 세 개의 다른 모델 시뮬레이션
    solar_ln = simulate_layernorm_weights(4096, seed=42)
    glm_ln = simulate_layernorm_weights(4096, seed=123)
    phi_ln = simulate_layernorm_weights(4096, seed=456)

    # 완전히 무작위 weights
    random_w = np.random.randn(4096).astype(np.float32)

    metrics = SimilarityMetrics()

    comparisons = [
        ("Solar vs GLM", solar_ln, glm_ln),
        ("GLM vs Phi", glm_ln, phi_ln),
        ("Solar vs Phi", solar_ln, phi_ln),
        ("Solar vs Random", solar_ln, random_w),
    ]

    print(f"{'Comparison':<20} {'Cosine Sim':>12} {'MAD':>12} {'Pearson':>12}")
    print("-" * 60)

    for name, w1, w2 in comparisons:
        cos = metrics.cosine_similarity(w1, w2)
        mad = metrics.mean_abs_diff(w1, w2)
        pearson = metrics.pearson_correlation(w1, w2)

        print(f"{name:<20} {cos:>12.4f} {mad:>12.4f} {pearson:>12.4f}")

    print()
    print("관찰 결과:")
    print("  - 모든 LayerNorm 비교에서 Cosine Similarity > 0.99")
    print("  - 이는 모두 1.0 근처에 분포하기 때문 (초기화 artifact)")
    print("  - MAD와 Pearson은 실제 차이를 더 잘 반영")
    print()


def demo_attention_comparison():
    """
    Attention Weight Shape 차이 시연

    Solar (64 heads) vs GLM (96 heads) 비교 불가
    """
    print("=" * 70)
    print("           Attention Weight Shape Comparison")
    print("=" * 70)
    print()

    hidden_size = 4096

    # Solar: 64 query heads
    solar_q_proj = np.random.randn(64 * 128, hidden_size).astype(np.float32)

    # GLM: 96 query heads
    glm_q_proj = np.random.randn(96 * 128, hidden_size).astype(np.float32)

    print(f"Solar Q projection shape: {solar_q_proj.shape}")
    print(f"GLM Q projection shape:   {glm_q_proj.shape}")
    print()

    result = compare_weights(solar_q_proj, glm_q_proj, "q_proj")

    print(f"Comparable: {result.comparable}")
    if not result.comparable:
        print(f"Shape mismatch: {result.shape_mismatch}")
        print()
        print("결론: Q/O projection은 shape이 달라 직접 비교 불가")
        print("      Solar(64 heads)와 GLM(96 heads)의 attention 구조가 다름")
    print()


def demo_moe_comparison():
    """
    MoE Expert Weight 차이 시연

    intermediate_size가 다름: 1280 vs 1408
    """
    print("=" * 70)
    print("           MoE Expert Weight Shape Comparison")
    print("=" * 70)
    print()

    hidden_size = 4096

    # Solar: moe_intermediate_size = 1280
    solar_expert_gate = np.random.randn(1280, hidden_size).astype(np.float32)
    solar_expert_up = np.random.randn(1280, hidden_size).astype(np.float32)
    solar_expert_down = np.random.randn(hidden_size, 1280).astype(np.float32)

    # GLM: moe_intermediate_size = 1408
    glm_expert_gate = np.random.randn(1408, hidden_size).astype(np.float32)
    glm_expert_up = np.random.randn(1408, hidden_size).astype(np.float32)
    glm_expert_down = np.random.randn(hidden_size, 1408).astype(np.float32)

    print(f"Solar expert gate shape: {solar_expert_gate.shape}")
    print(f"GLM expert gate shape:   {glm_expert_gate.shape}")
    print()

    result = compare_weights(solar_expert_gate, glm_expert_gate, "expert_gate")

    print(f"Comparable: {result.comparable}")
    if not result.comparable:
        print(f"Shape mismatch: {result.shape_mismatch}")
        print()
        print("결론: Expert FFN weights는 shape이 달라 직접 비교 불가")
        print("      Solar(1280)와 GLM(1408)의 intermediate size가 다름")
    print()


def generate_comparison_table():
    """
    Config 비교 테이블 생성
    """
    print("=" * 70)
    print("           Configuration Comparison Table")
    print("=" * 70)
    print()

    configs = {
        "hidden_size": (4096, 4096, "✓"),
        "num_hidden_layers": (48, 46, "✗"),
        "num_attention_heads": (64, 96, "✗"),
        "num_key_value_heads": (8, 8, "✓"),
        "head_dim": (128, 128, "✓"),
        "intermediate_size": (10240, 10944, "✗"),
        "moe_intermediate_size": (1280, 1408, "✗"),
        "n_routed_experts": (128, 128, "✓"),
        "n_shared_experts": (1, 1, "✓"),
        "num_experts_per_tok": (8, 8, "✓"),
        "max_position_embeddings": (131072, 131072, "✓"),
        "rope_theta": (1000000, 1000000, "✓"),
        "partial_rotary_factor": (1.0, 0.5, "✗"),
        "vocab_size": (196608, 151552, "✗"),
    }

    print(f"{'Parameter':<25} {'Solar':>12} {'GLM':>12} {'Match':>8}")
    print("-" * 60)

    same_count = 0
    diff_count = 0

    for param, (solar, glm, match) in configs.items():
        print(f"{param:<25} {str(solar):>12} {str(glm):>12} {match:>8}")
        if match == "✓":
            same_count += 1
        else:
            diff_count += 1

    print("-" * 60)
    print(f"Total: {same_count} same, {diff_count} different")
    print()


def main():
    parser = argparse.ArgumentParser(description="Weight comparison analysis")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    args = parser.parse_args()

    if args.demo:
        print("\n" + "=" * 70)
        print("      Solar Open 100B vs GLM-4.5-Air Weight Comparison Demo")
        print("=" * 70 + "\n")

        simulate_comparison_demo()
        demo_attention_comparison()
        demo_moe_comparison()
        generate_comparison_table()

        print("=" * 70)
        print("                         Summary")
        print("=" * 70)
        print()
        print("1. LayerNorm Cosine Similarity 한계:")
        print("   - 모든 모델이 0.99+ 유사도를 보임")
        print("   - 초기화 (≈1.0) artifact로 인한 false positive")
        print()
        print("2. Attention/Expert 비교 불가:")
        print("   - Q/O projection: shape 다름 (64 vs 96 heads)")
        print("   - Expert FFN: shape 다름 (1280 vs 1408)")
        print()
        print("3. 적절한 비교 방법:")
        print("   - Mean Absolute Difference (MAD)")
        print("   - Pearson Correlation")
        print("   - 동일 shape 가중치만 비교 (K, V, Router 등)")
        print()
    else:
        print("Use --demo for demonstration mode")
        print()
        print("For actual model comparison, you need:")
        print("  - 8x A100 80GB GPUs")
        print("  - transformers >= 4.54.0")
        print("  - Access to both models on HuggingFace")


if __name__ == "__main__":
    main()
