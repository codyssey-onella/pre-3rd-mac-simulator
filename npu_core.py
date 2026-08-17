"""Mini NPU 도메인 로직 — MAC 연산, 판정, 라벨 정규화, 성능 측정."""

from __future__ import annotations

import time
from enum import Enum

Matrix = list[list[float]]

EPSILON: float = 1e-9
PERFORMANCE_REPEAT_COUNT: int = 10


class Label(Enum):
    CROSS = "Cross"
    X = "X"
    UNDECIDED = "UNDECIDED"


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
    """위치별 곱셈 후 누적(MAC). NumPy 없이 반복문."""
    size: int = len(pattern)
    total: float = 0.0

    """
    싹 다 돌면서 둘 중 하나라도 0이 있으면 계산하지 않기?
    """
    for row_index in range(size):
        for col_index in range(size):
            total += pattern[row_index][col_index] * filter_matrix[row_index][col_index]
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
