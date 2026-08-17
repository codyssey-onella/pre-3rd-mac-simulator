"""Mini NPU 도메인 로직 — MAC 연산, 판정, 라벨 정규화, 성능 측정."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

Matrix = list[list[float]]

EPSILON: float = 1e-9
PERFORMANCE_REPEAT_COUNT: int = 10
# 3×3처럼 연산이 너무 짧으면 10회로는 오차가 커서, 보너스 비교만 반복을 늘린다.
OPTIMIZATION_REPEAT_COUNT: int = 20000


class Label(Enum):
    CROSS = "Cross"
    X = "X"
    UNDECIDED = "UNDECIDED"


class FilterType(Enum):
    """Java의 FilterType enum. 기하 공식으로 유효 인덱스만 뽑을 때 쓴다."""

    CROSS = "cross"
    X = "x"


SparseWeight = tuple[int, float]


@dataclass
class MacVariantBenchmark:
    # 1. 기존 2D 이중 루프 (mac)
    two_d_ms: float
    # 2. 1D로만 편 뒤 전칸 순회 (mac_1d)
    one_d_ms: float
    # 3. enum으로 인덱스만 뽑고, 1D 배열 두 개를 idx로 조회 (mac_indexmap)
    indexmap_ms: float
    # 4. (idx, filterValue) 튜플만 순회 (mac_sparse)
    sparse_ms: float


def get_matrix_size(matrix: Matrix) -> int:
    if not matrix:
        return 0
    return len(matrix)


def validate_matrix(matrix: Matrix, expected_size: int | None = None) -> str | None:
    """행·열 검증. 문제 없으면 None, 있으면 오류 메시지."""
    if not matrix:
        return "행렬이 비어 있습니다."
    row_count: int = len(matrix)
    if expected_size is not None and row_count != expected_size:
        return f"행 수 불일치: {row_count}행 (기대 {expected_size}행)"
    col_count: int = len(matrix[0])
    for row_index, row in enumerate(matrix):
        if len(row) != col_count:
            return f"열 수 불일치: {row_index + 1}행에 {len(row)}개 (기대 {col_count}개)"
    if expected_size is not None and col_count != expected_size:
        return f"열 수 불일치: {col_count}열 (기대 {expected_size}열)"
    return None


def mac(pattern: Matrix, filter_matrix: Matrix) -> float:
    """1. 기존 2D MAC. pattern[r][c] * filter[r][c] 이중 루프."""
    size: int = len(pattern)
    total: float = 0.0
    for row_index in range(size):
        for col_index in range(size):
            total += pattern[row_index][col_index] * filter_matrix[row_index][col_index]
    return total


def flatten_matrix(matrix: Matrix) -> list[float]:
    """N×N → 길이 N² 1차원 리스트. 시간 측정 구간 밖에서 호출한다."""
    flat: list[float] = []
    for row in matrix:
        for value in row:
            flat.append(value)
    return flat


def mac_1d(flat_pattern: list[float], flat_filter: list[float]) -> float:
    """2. 1차원으로만 편 MAC. 단일 루프, 칸은 N²개 전부."""
    total: float = 0.0
    length: int = len(flat_pattern)
    for index in range(length):
        total += flat_pattern[index] * flat_filter[index]
    return total


def create_shape_indices(filter_type: FilterType, size: int) -> list[int]:
    """3. 필터 배열을 훑지 않고, 모양 공식으로 1D 인덱스만 만든다.

    1D 위치 = row * size + col  (Java로 치면 2차원 배열을 한 줄로 편 오프셋)
    Cross: 가운데 열 + 가운데 행 (중심은 한 번만)
    X: 두 대각선 (중심은 한 번만)
    """
    indices: list[int] = []
    mid: int = size // 2

    if filter_type is FilterType.CROSS:
        for row_index in range(size):
            indices.append(row_index * size + mid)
        for col_index in range(size):
            if col_index == mid:
                continue
            indices.append(mid * size + col_index)
    else:
        # FilterType.X
        for row_index in range(size):
            indices.append(row_index * size + row_index)
            mirror: int = size - 1 - row_index
            if mirror != row_index:
                indices.append(row_index * size + mirror)

    return indices


def mac_indexmap(
    flat_pattern: list[float],
    flat_filter: list[float],
    indices: list[int],
) -> float:
    """3. 인덱스만 들고 pattern[idx] * filter[idx] — 리스트 조회 2번."""
    total: float = 0.0
    for index in indices:
        total += flat_pattern[index] * flat_filter[index]
    return total


def build_sparse_weights(flat_filter: list[float]) -> list[SparseWeight]:
    """4. 필터에서 0이 아닌 (idx, value)만 뽑는다. 전처리, 측정 제외."""
    weights: list[SparseWeight] = []
    for index, value in enumerate[float](flat_filter):
        if abs(value) > EPSILON:
            weights.append((index, value))
    return weights


def mac_sparse(flat_pattern: list[float], weights: list[SparseWeight]) -> float:
    """4. 튜플 (idx, filterValue). 패턴만 1번 조회하고 가중치는 로컬 변수."""
    total: float = 0.0
    for index, weight in weights:
        total += flat_pattern[index] * weight
    return total


def judge_cross_x(score_cross: float, score_x: float) -> Label:
    """Cross/X 필터 점수 비교 → 표준 판정 라벨."""
    if abs(score_cross - score_x) < EPSILON:
        return Label.UNDECIDED
    if score_cross > score_x:
        return Label.CROSS
    return Label.X


def judge_ab(score_a: float, score_b: float) -> str:
    """모드 1: 필터 A/B 점수 비교."""
    if abs(score_a - score_b) < EPSILON:
        return f"판정 불가 (|A-B| < {EPSILON})"
    if score_a > score_b:
        return "A"
    return "B"


def normalize_label(raw: str) -> Label | None:
    """JSON 라벨('+', 'x', 'cross' 등) → 표준 Label."""
    normalized: str = raw.strip().lower()
    if normalized in ("+", "cross"):
        return Label.CROSS
    if normalized == "x":
        return Label.X
    return None


def measure_mac_time_ms(pattern: Matrix, filter_matrix: Matrix) -> float:
    """MAC 연산 PERFORMANCE_REPEAT_COUNT회 평균 시간(ms). I/O 제외."""
    start: float = time.perf_counter()
    for _ in range(PERFORMANCE_REPEAT_COUNT):
        mac(pattern, filter_matrix)
    elapsed_seconds: float = time.perf_counter() - start
    return (elapsed_seconds / PERFORMANCE_REPEAT_COUNT) * 1000.0


def measure_mac_1d_time_ms(
    flat_pattern: list[float],
    flat_filter: list[float],
    repeat_count: int,
) -> float:
    start: float = time.perf_counter()
    for _ in range(repeat_count):
        mac_1d(flat_pattern, flat_filter)
    elapsed_seconds: float = time.perf_counter() - start
    return (elapsed_seconds / repeat_count) * 1000.0


def measure_mac_indexmap_time_ms(
    flat_pattern: list[float],
    flat_filter: list[float],
    indices: list[int],
    repeat_count: int,
) -> float:
    start: float = time.perf_counter()
    for _ in range(repeat_count):
        mac_indexmap(flat_pattern, flat_filter, indices)
    elapsed_seconds: float = time.perf_counter() - start
    return (elapsed_seconds / repeat_count) * 1000.0


def measure_mac_sparse_time_ms(
    flat_pattern: list[float],
    weights: list[SparseWeight],
    repeat_count: int,
) -> float:
    start: float = time.perf_counter()
    for _ in range(repeat_count):
        mac_sparse(flat_pattern, weights)
    elapsed_seconds: float = time.perf_counter() - start
    return (elapsed_seconds / repeat_count) * 1000.0


def measure_mac_2d_time_ms(
    pattern: Matrix,
    filter_matrix: Matrix,
    repeat_count: int,
) -> float:
    start: float = time.perf_counter()
    for _ in range(repeat_count):
        mac(pattern, filter_matrix)
    elapsed_seconds: float = time.perf_counter() - start
    return (elapsed_seconds / repeat_count) * 1000.0


def create_zero_matrix(size: int) -> Matrix:
    """N×N 영행렬. 행마다 새 리스트를 만든다."""
    matrix: Matrix = []
    for row_index in range(size):
        row: list[float] = []
        for col_index in range(size):
            row.append(0.0)
        matrix.append(row)
    return matrix


def create_cross_pattern(size: int) -> Matrix:
    """N×N 십자가(Cross) 패턴 생성."""
    mid: int = size // 2
    matrix: Matrix = create_zero_matrix(size)
    for row_index in range(size):
        matrix[row_index][mid] = 1.0
        matrix[mid][row_index] = 1.0
    return matrix


def create_x_pattern(size: int) -> Matrix:
    """N×N X 패턴 생성."""
    matrix: Matrix = create_zero_matrix(size)
    for row_index in range(size):
        matrix[row_index][row_index] = 1.0
        matrix[row_index][size - 1 - row_index] = 1.0
    return matrix


def extract_size_from_pattern_key(key: str) -> int | None:
    """patterns 키 'size_{N}_{idx}' 에서 N 추출."""
    parts: list[str] = key.split("_")
    if len(parts) < 3 or parts[0] != "size":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def format_performance_row(size: int, avg_ms: float) -> str:
    op_count: int = size * size
    return f"{size}×{size:<10} {avg_ms:>10.3f}       {op_count}"


def benchmark_mac_variants(
    pattern: Matrix,
    filter_matrix: Matrix,
    filter_type: FilterType,
    repeat_count: int = OPTIMIZATION_REPEAT_COUNT,
) -> MacVariantBenchmark:
    """전처리 후 1~4 MAC만 측정. flatten/인덱스/튜플 생성은 시간에서 뺀다."""
    size: int = len(pattern)
    # 2·3·4 공통 재료: 2D → 1D (측정 제외)
    flat_pattern: list[float] = flatten_matrix(pattern)
    flat_filter: list[float] = flatten_matrix(filter_matrix)
    # 3 재료: FilterType 공식으로 유효 1D 인덱스만 (측정 제외)
    indices: list[int] = create_shape_indices(filter_type, size)
    # 4 재료: 필터에서 0 아닌 (idx, value) (측정 제외)
    weights: list[SparseWeight] = build_sparse_weights(flat_filter)

    return MacVariantBenchmark(
        two_d_ms=measure_mac_2d_time_ms(pattern, filter_matrix, repeat_count),  # 1
        one_d_ms=measure_mac_1d_time_ms(flat_pattern, flat_filter, repeat_count),  # 2
        indexmap_ms=measure_mac_indexmap_time_ms(
            flat_pattern, flat_filter, indices, repeat_count
        ),  # 3
        sparse_ms=measure_mac_sparse_time_ms(flat_pattern, weights, repeat_count),  # 4
    )
