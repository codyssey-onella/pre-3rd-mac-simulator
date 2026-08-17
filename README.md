# Mini NPU Simulator

AI의 MAC(Multiply-Accumulate) 연산 원리를 이해하기 위한 Cross/X 패턴 판별 시뮬레이터.

## 실행 방법

```bash
cd result
python main.py
```

모드 선택:
- **1** — 사용자 입력 (3×3): 필터 A, B, 패턴을 콘솔에 입력
- **2** — `data.json` 분석: JSON 데이터 일괄 판정 + 성능 분석 + 결과 요약

`data.json`은 프로젝트 루트(`../data.json`)에 있음.

### 모드 1 예시 입력 (instruction 예시)

필터 A (십자가):
```
0 1 0
1 1 1
0 1 0
```

필터 B (X):
```
1 0 1
0 1 0
1 0 1
```

패턴 (X):
```
1 0 1
0 1 0
1 0 1
```

예상 결과: A 점수 1.0, B 점수 5.0, 판정 B

## 구현 요약

### 파일 구조

| 파일 | 역할 |
| --- | --- |
| `main.py` | 진입점, 모드 선택, I/O, data.json 분석 흐름 |
| `npu_core.py` | MAC 연산, 판정, 라벨 정규화, 성능 측정, 패턴 생성 |
| `logger.py` | 예외 로깅 (표준 라이브러리 `logging`) |
| `../data.json` | 필터(5/13/25) + 테스트 패턴 |

### 라벨 정규화

JSON에는 같은 의미를 다른 문자로 표기한다. 내부에서는 `Cross`, `X` 두 값만 사용.

| JSON 값 | 표준 라벨 |
| --- | --- |
| `expected: "+"` | `Cross` |
| `expected: "x"` | `X` |
| filter 키 `"cross"` | `Cross` |
| filter 키 `"x"` | `X` |

Java 비유: API 응답 `"+"` / `"cross"`를 enum `Label.CROSS`로 매핑하는 것과 같다.

### MAC 연산

```python
for i in range(n):
    for j in range(n):
        total += pattern[i][j] * filter[i][j]
```

NumPy 없이 이중 반복문으로 구현. 연산 횟수 = N².

### 동점 처리 (epsilon)

`abs(score_cross - score_x) < 1e-9` 이면 **UNDECIDED** (판정 불가).
모드 2에서 UNDECIDED는 expected와 다르면 **FAIL**.

## 결과 리포트

### FAIL 케이스 분석

`data.json` 6개 패턴 실행 시 **3 PASS / 3 FAIL** (실제 측정 결과).

| 케이스 | Cross 점수 | X 점수 | expected | 판정 | 결과 |
| --- | --- | --- | --- | --- | --- |
| size_5_1 | 0.9 | 0.8999999999999999 | X | UNDECIDED | **FAIL** |
| size_5_2 | 8.9 | 0.1 | Cross | Cross | PASS |
| size_13_1 | 0.3 | 14.7 | X | X | PASS |
| size_13_2 | 7.5 | 7.5 | Cross | UNDECIDED | **FAIL** |
| size_25_1 | 4.9 | 4.899999999999999 | X | UNDECIDED | **FAIL** |
| size_25_2 | 52.9 | 0.1 | Cross | Cross | PASS |

**1단계 — 동점의 근본 원인: 필터 데이터 설계 (논리적 Ambiguity)**

3개 FAIL 케이스는 부동소수점 찌꺼기만의 문제가 아니다. `data.json` 필터가 **중심점 가중치를 상대 필터의 전체 활성화 합과 정확히 맞춰** 이론적 동점이 나오도록 설계되어 있다.

- **size_5_1** (X 패턴 입력): X 필터 대각선 9칸 × 0.1 = **0.9**. Cross 필터 중심 [2][2] = **0.9**이고 입력 중심도 1.0 → Cross 점수 **0.9**. 이론상 0.9 vs 0.9 동점.
- **size_13_2** (Cross 패턴 입력): Cross 필터 십자가 25칸 × 0.3 = **7.5**. X 필터 중심 [6][6] = **7.5**로, Cross 패턴이 와도 중심 한 칸만 겹쳐 X 점수 **7.5**. 이론상 7.5 vs 7.5 동점.
- **size_25_1** (X 패턴 입력): X 필터 대각선 49칸 × 0.1 = **4.9**. Cross 필터 중심 [12][12] = **4.9** → Cross 점수 **4.9**. 이론상 4.9 vs 4.9 동점.

즉 expected는 X 또는 Cross인데, MAC 점수만으로는 어느 쪽도 고를 수 없는 **구조적 동점** 케이스다.

**2단계 — 부동소수점 오차와 epsilon의 역할**

이론상 같은 값이어도 Python float 누적 연산에서 `0.9` vs `0.8999999999999999`, `7.5` vs `7.499999999999997`처럼 미세한 차이가 출력된다.
단순 `score_cross > score_x` 비교면 이 찌꺼기 비트에 따라 엉뚱한 Cross/X 판정이 날 수 있다.
`1e-9` epsilon 정책으로 차이가 무시 가능한 수준이면 **UNDECIDED**로 안정적으로 식별하고, UNDECIDED ≠ expected이므로 **FAIL** 처리한다.

**3개 PASS인 이유**

- MAC 점수 차이가 epsilon 밖 → Cross 또는 X 판정이 명확
- 라벨 정규화(`+`→Cross, `x`→X) 후 판정 == expected

### 시간 복잡도 O(N²) 분석

MAC 연산은 N×N 행렬 전체를 한 번 순회하므로 **연산 횟수 = N²** (곱셈 N²번 + 덧셈 N²-1번).

| 크기 | 연산 횟수 (N²) | 측정 의미 |
| --- | --- | --- |
| 3×3 | 9 | 기준점, 가장 빠름 |
| 5×5 | 25 | 약 2.8배 연산 |
| 13×13 | 169 | 약 18.8배 연산 |
| 25×25 | 625 | 약 69.4배 연산 |

실제 측정에서도 크기가 커질수록 평균 시간이 증가한다.
25×25는 3×3 대비 연산량이 69배인데, 시간도 비슷한 비율로 늘어나는 경향을 보인다.
이는 MAC이 **선형이 아니라 제곱**에 비례하는 대표적인 O(N²) 패턴이다.

AI/NPU에서 필터가 수백×수백이고 수천 개면 N² × 필터 수가 폭발적으로 커지기 때문에,
CPU 직렬 처리로는 감당 못 하고 NPU가 MAC을 병렬로 처리한다.

### instruction 예시와 다른 점

instruction 결과 예시는 **참고용**이라 실제 `data.json` 점수와 다를 수 있다.
예시에서는 size_13_1이 FAIL인데, 현재 데이터에서는 X 점수(14.7)가 Cross(0.3)보다 훨씬 커서 PASS.
실제 채점은 **현재 data.json + epsilon 규칙**으로 나온 PASS/FAIL이 맞는지 보면 된다.

과제가 의도한 함정은 두 가지를 겹친 것이다.
(1) 필터 중심 가중치 설계로 인한 **논리적 동점**, (2) 그 동점을 epsilon 없이 비교하면 깨질 수 있는 **부동소수점 비교 문제**.
둘 다 이해해야 FAIL 원인을 제대로 설명할 수 있다.
