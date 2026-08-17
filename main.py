"""Mini NPU Simulator — MAC 연산 기반 Cross/X 패턴 판별 시뮬레이터."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from logger import get_logger
from npu_core import (
    FilterType,
    Label,
    MacVariantBenchmark,
    Matrix,
    OPTIMIZATION_REPEAT_COUNT,
    PERFORMANCE_REPEAT_COUNT,
    benchmark_mac_variants,
    create_cross_pattern,
    create_x_pattern,
    extract_size_from_pattern_key,
    format_performance_row,
    get_matrix_size,
    judge_ab,
    judge_cross_x,
    mac,
    measure_mac_time_ms,
    normalize_label,
    validate_matrix,
)

logger = get_logger()

MATRIX_SIZE_3: int = 3
DATA_JSON_PATH: Path = Path(__file__).resolve().parent / "data.json"
ERROR_RETRY_MESSAGE: str = "에러가 발생했습니다. 모드를 다시 선택해주세요."


class Mode(Enum):
    USER_INPUT = (1, "사용자 입력 (3x3)")
    DATA_JSON = (2, "data.json 분석")

    def __init__(self, number: int, label: str) -> None:
        self.number: int = number
        self.label: str = label

    @classmethod
    def from_number(cls, number: int) -> Mode | None:
        for mode in cls:
            if mode.number == number:
                return mode
        return None

    @classmethod
    def print_all(cls) -> None:
        for mode in cls:
            print(f"{mode.number}. {mode.label}")


@dataclass
class PatternResult:
    case_id: str
    cross_score: float | None
    x_score: float | None
    judgment: Label | None
    expected: Label | None
    passed: bool
    fail_reason: str | None

def main() -> None:
    run()

def run() -> None:
    print_banner()
    while True:
        try:
            mode: Mode = read_mode()
            if mode is Mode.USER_INPUT:
                run_user_input_mode()
            elif mode is Mode.DATA_JSON:
                run_data_json_mode()
            print("\n판정이 완료되었습니다. 모드를 다시 선택해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
            return
        except EOFError:
            print("\n입력이 종료되어 프로그램을 종료합니다.")
            return
        except Exception:
            logger.exception("실행 중 예상치 못한 에러가 발생했습니다")
            print(ERROR_RETRY_MESSAGE)

def read_mode() -> Mode:
    invalid_message: str = "잘못된 입력입니다. 1 또는 2를 입력하세요."
    while True:
        print("\n[모드 선택]")
        Mode.print_all()
        raw: str = read_line("선택: ")
        stripped: str = raw.strip()
        if stripped == "":
            print(invalid_message)
            continue
        try:
            value: int = int(stripped)
        except ValueError:
            print(invalid_message)
            continue
        mode: Mode | None = Mode.from_number(value)
        if mode is None:
            print(invalid_message)
            continue
        return mode

def read_line(prompt: str = "") -> str:
    return input(prompt)

def parse_matrix_row(raw: str, size: int) -> list[float] | None:
    parts: list[str] = raw.strip().split()
    if len(parts) != size:
        return None
    row: list[float] = []
    for part in parts:
        try:
            row.append(float(part))
        except ValueError:
            return None
    return row


def read_matrix(size: int, title: str) -> Matrix:
    print(f"\n{title} ({size}줄 입력, 공백 구분)")
    while True:
        rows: Matrix = []
        for _ in range(size):
            raw: str = read_line("")
            parsed: list[float] | None = parse_matrix_row(raw, size)
            if parsed is None:
                print(
                    f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
                )
                rows.clear()
                break
            rows.append(parsed)
        if len(rows) == size:
            return rows


def format_matrix(matrix: Matrix) -> str:
    lines: list[str] = []
    for row in matrix:
        line = " ".join(str(value) for value in row)
        lines.append(line)
    return "\n".join(lines)


def print_performance_table(
    sizes: list[int], filters_by_size: dict[int, dict[str, Matrix]]
) -> None:
    print("\n#----------------------------------------")
    print(f"# [3] 성능 분석 (평균/{PERFORMANCE_REPEAT_COUNT}회)")
    print("#----------------------------------------")
    print("크기       평균 시간(ms)    연산 횟수")
    print("-------------------------------------")
    for size in sizes:
        if size == MATRIX_SIZE_3:
            cross_filter: Matrix = create_cross_pattern(size)
            x_filter: Matrix = create_x_pattern(size)
        else:
            size_filters: dict[str, Matrix] | None = filters_by_size.get(size)
            if size_filters is None:
                continue
            cross_filter = size_filters["cross"]
            x_filter = size_filters["x"]
        pattern: Matrix = create_cross_pattern(size)
        cross_time: float = measure_mac_time_ms(pattern, cross_filter)
        x_time: float = measure_mac_time_ms(pattern, x_filter)
        avg_ms: float = (cross_time + x_time) / 2.0
        print(format_performance_row(size, avg_ms))

    # 보너스 - 단계적 개선 별 성능 비교
    print_optimization_table(sizes, filters_by_size)


def resolve_filters_for_size(
    size: int, filters_by_size: dict[int, dict[str, Matrix]]
) -> tuple[Matrix, Matrix] | None:
    if size == MATRIX_SIZE_3:
        return create_cross_pattern(size), create_x_pattern(size)
    size_filters: dict[str, Matrix] | None = filters_by_size.get(size)
    if size_filters is None:
        return None
    return size_filters["cross"], size_filters["x"]


def print_optimization_table(
    sizes: list[int], filters_by_size: dict[int, dict[str, Matrix]]
) -> None:
    """보너스 표. 열 순서 = 개선 단계 1→4.

    1. two_d    = 기존 2D mac()
    2. one_d    = flatten 후 전칸 1D mac_1d()
    3. indexmap = FilterType으로 인덱스만 뽑고, 1D 두 배열을 idx로 조회 mac_indexmap()
    4. sparse   = (idx, filterValue) 튜플 mac_sparse()
    """
    print("\n#----------------------------------------")
    print(f"# [보너스] MAC 4방식 비교 (평균/{OPTIMIZATION_REPEAT_COUNT}회, 전처리 제외)")
    print("#----------------------------------------")
    print("1. 기존 2D     2. 1D flatten     3. enum IndexMap     4. (idx, value) 튜플")
    print("크기     1.2D(ms)   2.1D(ms)   3.Idx(ms)   4.Tuple(ms)")
    print("-------------------------------------------------------")
    for size in sizes:
        resolved: tuple[Matrix, Matrix] | None = resolve_filters_for_size(
            size, filters_by_size
        )
        if resolved is None:
            continue
        cross_filter, x_filter = resolved
        pattern: Matrix = create_cross_pattern(size)
        cross_bench: MacVariantBenchmark = benchmark_mac_variants(
            pattern, cross_filter, FilterType.CROSS
        )
        x_bench: MacVariantBenchmark = benchmark_mac_variants(
            pattern, x_filter, FilterType.X
        )
        # Cross/X 필터 시간을 평균해서 한 줄로 출력
        two_d: float = (cross_bench.two_d_ms + x_bench.two_d_ms) / 2.0  # 1 기존 2D
        one_d: float = (cross_bench.one_d_ms + x_bench.one_d_ms) / 2.0  # 2 1D 전칸
        indexmap: float = (
            cross_bench.indexmap_ms + x_bench.indexmap_ms
        ) / 2.0  # 3 인덱스+1D 조회
        sparse: float = (cross_bench.sparse_ms + x_bench.sparse_ms) / 2.0  # 4 튜플
        print(
            f"{size}×{size:<4} "
            f"{two_d:>9.4f}  "  # 1
            f"{one_d:>9.4f}  "  # 2
            f"{indexmap:>9.4f}  "  # 3
            f"{sparse:>10.4f}"  # 4
        )


def run_user_input_mode() -> None:
    print("\n#----------------------------------------")
    print("# [1] 필터 입력")
    print("#----------------------------------------")
    filter_a: Matrix = read_matrix(MATRIX_SIZE_3, "필터 A")
    print("필터 A 저장 완료")
    filter_b: Matrix = read_matrix(MATRIX_SIZE_3, "필터 B")
    print("필터 B 저장 완료")

    print("\n#----------------------------------------")
    print("# [2] 패턴 입력")
    print("#----------------------------------------")
    pattern: Matrix = read_matrix(MATRIX_SIZE_3, "패턴")

    score_a: float = mac(pattern, filter_a)
    score_b: float = mac(pattern, filter_b)
    avg_time_ms: float = (
        measure_mac_time_ms(pattern, filter_a) + measure_mac_time_ms(pattern, filter_b)
    ) / 2.0
    judgment: str = judge_ab(score_a, score_b)

    print("\n#----------------------------------------")
    print("# [3] MAC 결과")
    print("#----------------------------------------")
    print(f"A 점수: {score_a}")
    print(f"B 점수: {score_b}")
    print(f"연산 시간(평균/{PERFORMANCE_REPEAT_COUNT}회): {avg_time_ms:.3f} ms")
    print(f"판정: {judgment}")
    print_optimization_table([MATRIX_SIZE_3], {})


def load_data_json() -> dict:
    with DATA_JSON_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_filters(data: dict) -> tuple[dict[int, dict[str, Matrix]], list[str]]:
    filters_raw: dict = data.get("filters", {})
    filters_by_size: dict[int, dict[str, Matrix]] = {}
    errors: list[str] = []

    for size_key, filter_pair in filters_raw.items():
        if not size_key.startswith("size_"):
            errors.append(f"필터 키 형식 오류: {size_key}")
            continue
        try:
            size: int = int(size_key.split("_")[1])
        except (IndexError, ValueError):
            errors.append(f"필터 크기 추출 실패: {size_key}")
            continue

        cross_raw: Matrix | None = filter_pair.get("cross")
        x_raw: Matrix | None = filter_pair.get("x")
        if cross_raw is None or x_raw is None:
            errors.append(f"{size_key}: cross 또는 x 필터 누락")
            continue

        cross_error: str | None = validate_matrix(cross_raw, size)
        x_error: str | None = validate_matrix(x_raw, size)
        if cross_error or x_error:
            errors.append(f"{size_key}: 필터 크기 검증 실패")
            continue

        filters_by_size[size] = {"cross": cross_raw, "x": x_raw}

    return filters_by_size, errors


def analyze_pattern(
    case_id: str,
    pattern_data: dict,
    filters_by_size: dict[int, dict[str, Matrix]],
) -> PatternResult:
    size: int | None = extract_size_from_pattern_key(case_id)
    if size is None:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=None,
            passed=False,
            fail_reason="케이스 키 형식 오류 (size_{N}_{idx} 필요)",
        )

    input_raw: Matrix | None = pattern_data.get("input")
    expected_raw: str | None = pattern_data.get("expected")
    if input_raw is None or expected_raw is None:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=None,
            passed=False,
            fail_reason="input 또는 expected 필드 누락",
        )

    expected_label: Label | None = normalize_label(expected_raw)
    if expected_label is None:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=None,
            passed=False,
            fail_reason=f"expected 라벨 정규화 실패: {expected_raw}",
        )

    size_filters: dict[str, Matrix] | None = filters_by_size.get(size)
    if size_filters is None:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=expected_label,
            passed=False,
            fail_reason=f"size_{size} 필터 없음",
        )

    pattern_error: str | None = validate_matrix(input_raw, size)
    if pattern_error:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=expected_label,
            passed=False,
            fail_reason=f"패턴 크기 불일치: {pattern_error}",
        )

    cross_filter: Matrix = size_filters["cross"]
    x_filter: Matrix = size_filters["x"]
    if get_matrix_size(cross_filter) != size or get_matrix_size(x_filter) != size:
        return PatternResult(
            case_id=case_id,
            cross_score=None,
            x_score=None,
            judgment=None,
            expected=expected_label,
            passed=False,
            fail_reason="필터와 패턴 크기 불일치",
        )

    cross_score: float = mac(input_raw, cross_filter)
    x_score: float = mac(input_raw, x_filter)
    judgment: Label = judge_cross_x(cross_score, x_score)
    passed: bool = judgment == expected_label

    fail_reason: str | None = None
    if not passed:
        if judgment == Label.UNDECIDED:
            fail_reason = "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
        else:
            fail_reason = f"판정 {judgment.value} ≠ expected {expected_label.value}"

    return PatternResult(
        case_id=case_id,
        cross_score=cross_score,
        x_score=x_score,
        judgment=judgment,
        expected=expected_label,
        passed=passed,
        fail_reason=fail_reason,
    )


def run_data_json_mode() -> None:
    data: dict = load_data_json()
    filters_by_size, filter_errors = load_filters(data)

    print("\n#----------------------------------------")
    print("# [1] 필터 로드")
    print("#----------------------------------------")
    for size in filters_by_size:
        print(f"✓ size_{size:<3} 필터 로드 완료 (Cross, X)")
    for error in filter_errors:
        print(f"✗ {error}")

    patterns_raw: dict = data.get("patterns", {})
    results: list[PatternResult] = []

    print("\n#----------------------------------------")
    print("# [2] 패턴 분석 (라벨 정규화 적용)")
    print("#----------------------------------------")
    for case_id in patterns_raw:
        result: PatternResult = analyze_pattern(
            case_id, patterns_raw[case_id], filters_by_size
        )
        results.append(result)
        print(f"--- {case_id} ---")
        # 점수를 구하지 못한 경우
        if result.fail_reason and result.cross_score is None:
            print(f"FAIL ({result.fail_reason})")
            continue

        # 점수가 있으면 print하고 결과 출력
        print(f"Cross 점수: {result.cross_score}")
        print(f"X 점수: {result.x_score}")
        status: str = "PASS" if result.passed else "FAIL"
        extra: str = ""
        if not result.passed and result.fail_reason:
            extra = f" ({result.fail_reason})"
        print(
            f"판정: {result.judgment.value} | expected: {result.expected.value} | {status}{extra}"
        )

    performance_sizes: list[int] = [MATRIX_SIZE_3, *filters_by_size]
    print_performance_table(performance_sizes, filters_by_size)

    total: int = len(results)
    passed_count: int = sum(1 for result in results if result.passed)
    failed_count: int = total - passed_count

    print("\n#----------------------------------------")
    print("# [4] 결과 요약")
    print("#----------------------------------------")
    print(f"총 테스트: {total}개")
    print(f"통과: {passed_count}개")
    print(f"실패: {failed_count}개")

    failed_cases: list[PatternResult] = [result for result in results if not result.passed]
    if failed_cases:
        print("\n실패 케이스:")
        for result in failed_cases:
            reason: str = result.fail_reason or "알 수 없음"
            print(f"- {result.case_id}: {reason}")
    print(
        '\n(상세 원인 분석 및 복잡도 설명은 README.md의 "결과 리포트" 섹션에 작성)'
    )


def print_banner() -> None:
    print("=== Mini NPU Simulator ===")


if __name__ == "__main__":
    main()
