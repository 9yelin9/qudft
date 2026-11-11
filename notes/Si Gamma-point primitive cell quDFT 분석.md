---
created: 2026-11-06T15:45:18+09:00
updated: 2026-11-11T16:13:55+09:00
tags:
  - qudft
  - au-swap
  - silicon
  - gamma-point
  - pauli-grouping
  - fixed-occupation
related:
  - "[[Structured AO binary encoding]]"
  - "[[quDFT Ansatz 구성 규칙]]"
  - "[[quDFT occupation 계산 방식]]"
---

# Si Gamma-point primitive cell quDFT 분석

## 목차

- [[#시스템 구성|시스템 구성]]
- [[#핵심 결과|핵심 결과]]
- [[#IonQ simulator optimization 진행 현황|IonQ simulator optimization 진행 현황]]
- [[#최소 Pauli 수 계산 조건|최소 Pauli 수 계산 조건]]
- [[#최소 Pauli 수 계산 결과|최소 Pauli 수 계산 결과]]
- [[#에너지 계산에서 확인된 사실|에너지 계산에서 확인된 사실]]
- [[#최소 P4 회로의 물리적 해석|최소 P4 회로의 물리적 해석]]
	- [[#최소 P4 회로의 물리적 해석#서로 다른 원자의 같은 orbital mixing|서로 다른 원자의 같은 orbital mixing]]
	- [[#최소 P4 회로의 물리적 해석#4개의 Pauli string이 필요한 이유|4개의 Pauli string이 필요한 이유]]
	- [[#최소 P4 회로의 물리적 해석#물리적 의미와 적용 범위|물리적 의미와 적용 범위]]
- [[#Density matrix 측정과 Pauli grouping|Density matrix 측정과 Pauli grouping]]
	- [[#Density matrix 측정과 Pauli grouping#측정 대상|측정 대상]]
	- [[#Density matrix 측정과 Pauli grouping#Group 구성|Group 구성]]
	- [[#Density matrix 측정과 Pauli grouping#측정 circuit 수|측정 circuit 수]]
	- [[#Density matrix 측정과 Pauli grouping#Parameter-shift gradient를 포함한 측정 비용|Parameter-shift gradient를 포함한 측정 비용]]
- [[#IonQ QPU dry-run 비용: GC와 QWC 비교|IonQ QPU dry-run 비용: GC와 QWC 비교]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#계산 조건|계산 조건]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#Grouping별 측정 규모|Grouping별 측정 규모]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#Shot 수와 dry-run 예상 비용|Shot 수와 dry-run 예상 비용]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#확인된 사실|확인된 사실]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#해석과 적용 범위|해석과 적용 범위]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#재현 명령과 원본 데이터|재현 명령과 원본 데이터]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#원격 IonQ simulator의 시간 병목|원격 IonQ simulator의 시간 병목]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#측정 비용 절감 방향|측정 비용 절감 방향]]
	- [[#IonQ QPU dry-run 비용: GC와 QWC 비교#구현과 실행|구현과 실행]]
- [[#원본 계산 데이터|원본 계산 데이터]]

## 시스템 구성

- 물질: primitive-cell Si
- 원자 수: 2
- basis: `gth-szv`
- AO 수: 8
- k-grid: `1x1x1` (Gamma point)
- AO encoding: 3 qubit structured AO encoding
- PennyLane wire 0: atom index
- PennyLane wire 1, 2: local AO index

Computational basis는 다음과 같이 분해된다.

$$
|q_0q_1q_2\rangle
=
|\mathrm{atom}\rangle
\otimes
|\mathrm{local\ AO}\rangle
$$

두 원자가 각각 4개의 AO를 가지므로 8개 computational-basis state가 모두 실제
AO에 대응하고 unused state는 없다.

## 핵심 결과

Primitive-cell Si의 Gamma-point fixed-occupation 계산에서 PySCF ground-state
에너지를 재현하는 최소 active Pauli 수는

$$
\boxed{N_p^{\min}=4}
$$

이다. 최소 active set은 다음과 같다.

```text
YZZ, YIZ, YZI, YII
```

## IonQ simulator optimization 진행 현황

아래 그래프는 2026-11-11T16:07:50+09:00 시점의 4개 독립 실행을
iteration별로 표시한 snapshot이다. 각 점은 1000 shots로 평가한 한
optimization iteration의 total energy이며, 회색 점선은 PySCF
zero-temperature 기준 에너지다.

![[fig/Si_ionq_simulator_resume_progress.png]]

아래 확대 그래프는 각 실행에서 가장 최근에 완료된 10개 iteration만
표시한다. 실행별 iteration 범위가 다르므로 2×2 panel로 분리했으며, 모든
panel의 y축 범위는 동일하다.

![[fig/Si_ionq_simulator_resume_progress_last10.png]]

$$
E_{\mathrm{PySCF,GS}}
=
-7.110403267952345\ E_h
$$

| 실행 | 마지막 iteration | $E_{\mathrm{tot}}$ ($E_h$) | `grad_theta` | `delta_efree` |
|---|---:|---:|---:|---:|
| Run 1 | 161 | -7.008129 | 0.003549 | 0.004435 |
| Run 2 | 180 | -7.005311 | 0.003084 | 0.003562 |
| Run 3 | 119 | -6.999601 | 0.017425 | 0.003346 |
| Run 4 | 164 | -7.000808 | 0.025710 | 0.002542 |

4개 실행은 약 $-7.00\ E_h$ 부근까지 공통적으로 내려간 뒤 shot noise가
지배적인 plateau를 보인다. Snapshot 시점에는 `grad_theta`와
`delta_efree`가 모두 $10^{-4}$보다 작아야 하는 resume 수렴 조건을 만족한
실행은 없다. 그래프는 다음 명령으로 현재 HDF5 history에서 다시 생성할 수
있다.

```bash
env MPLCONFIGDIR=/tmp/matplotlib-cache \
  python3 plot_ionq_simulator_progress.py
```

## 최소 Pauli 수 계산 조건

- 물질: primitive-cell Si
- k-grid: `1x1x1` (Gamma point)
- occupation: `[2, 2, 2, 2, 0, 0, 0, 0]`로 고정
- ansatz: 현재 명칭 `au-swap`에 해당하는 fixed-size adaptive ansatz
- active Pauli 수: 각 실행 중 고정
- 구조 변경: 에너지 평가로 승인하는 1-to-1 Pauli swap
- optimizer tolerance: `1e-5`
- 최대 iteration: `500`
- PySCF zero-temperature 기준 에너지:

$$
E_{\mathrm{PySCF,GS}}=-7.110403267952345\ E_h
$$

## 최소 Pauli 수 계산 결과

| $N_p$ | 최종 에너지 (Ha) | 오차 (Ha) | 오차 (eV) | Iteration | 수렴 | Swap | 최종 active Pauli strings |
|---:|---:|---:|---:|---:|:---:|---:|---|
| 1 | -6.916114409907 | +1.942888580e-01 | +5.286869 | 230 | O | 0 | `YZZ` |
| 2 | -6.916114409902 | +1.942888581e-01 | +5.286869 | 336 | O | 1 | `YZZ`, `YII` |
| 3 | -6.916114409807 | +1.942888581e-01 | +5.286869 | 384 | O | 1 | `YIZ`, `YZI`, `YII` |
| 4 | -7.110403268080 | -1.277040695e-10 | 약 0 | 230 | O | 0 | `YZZ`, `YIZ`, `YZI`, `YII` |
| 5 | -7.110403267998 | -4.566924616e-11 | 약 0 | 263 | O | 0 | `YZZ`, `YIZ`, `YZI`, `YII`, `YIX` |
| 6 | -7.110403268059 | -1.063744648e-10 | 약 0 | 247 | O | 0 | `YZZ`, `YIZ`, `YZI`, `YII`, `YIX`, `YXI` |
| 7 | -7.110403268010 | -5.743938658e-11 | 약 0 | 256 | O | 0 | `YZZ`, `YIZ`, `YZI`, `YII`, `YIX`, `YXI`, `YXX` |

## 에너지 계산에서 확인된 사실

1. P1-P3은 gradient 수렴 판정을 만족했지만 PySCF 기준보다 약
   $0.194289\ E_h$ 높은 동일한 stationary energy에 도달했다.
2. P2와 P3에서는 1-to-1 swap이 각각 한 번 승인됐지만 최종 에너지를
   개선하지 못했다.
3. P4-P7은 PySCF ground-state 에너지를 약 $10^{-10}\ E_h$ 이하 오차로
   재현했다.
4. 따라서 이 계산 조건에서 최소 active Pauli 수는 P4이다.

## 최소 P4 회로의 물리적 해석

다음 내용은 계산 결과에 기반한 구조적 해석이다.

P4 generator는

$$
Y\otimes\{II,IZ,ZI,ZZ\}
$$

형태다. Atom wire에는 `Y`가 있고 두 local-AO wire에는 $I/Z$의 4개 조합이
있다.

### 서로 다른 원자의 같은 orbital mixing

Structured AO encoding에서 basis state를

$$
|a,\ell\rangle
=|a\rangle_{\mathrm{atom}}\otimes|\ell\rangle_{\mathrm{local\ AO}}
$$

로 쓸 수 있다. 여기서 $a\in\{A,B\}$는 원자 index이고
$\ell\in\{0,1,2,3\}$는 원자 내부의 local AO index이다.

P4의 4개 Pauli string은 atom wire에만 $Y$가 있고 local-AO wire에는
$I$ 또는 $Z$만 있다. 따라서 atom bit만 반전하고 local AO index는 보존한다.
각 generator가 연결하는 상태는

$$
|A,\ell\rangle\longleftrightarrow|B,\ell\rangle
$$

이다. 즉, P4가 제공하는 핵심 자유도는 서로 다른 두 Si 원자의 **같은 local
orbital끼리의 mixing**이다. 4개의 local channel이 각각 $s,p_x,p_y,p_z$에
대응하는 순서라면 이는 $s$-$s$, $p_x$-$p_x$, $p_y$-$p_y$,
$p_z$-$p_z$ 원자 간 mixing을 뜻한다. 반대로 서로 다른 local orbital을
연결하는 $s$-$p$ 또는 $p_x$-$p_y$와 같은 mixing은 P4에 포함되지 않는다.

### 4개의 Pauli string이 필요한 이유

Local-AO wire의

$$
II,\quad IZ,\quad ZI,\quad ZZ
$$

는 orbital index를 바꾸지 않고 4개의 local AO에 서로 다른 부호 패턴을
부여한다. Local bit string을 $\ell=(\ell_1,\ell_0)$라 하면, 특정 local
AO만 선택하는 atom-mixing generator는

$$
Y_{\mathrm{atom}}\otimes|\ell\rangle\langle\ell|
=\frac{1}{4}\sum_{a,b\in\{0,1\}}
(-1)^{a\ell_1+b\ell_0}
Y_{\mathrm{atom}}\otimes Z_1^a Z_0^b
$$

로 표현된다. 따라서 `YII`, `YIZ`, `YZI`, `YZZ`의 4개 방향을 모두
사용하면 4개의 local orbital의 원자 간 mixing angle을 서로 독립적으로 만들
수 있다. P1-P3에서는 부호 패턴의 수가 부족하여 일부 orbital의 mixing
amplitude가 서로 묶인다. P4에서 비로소 4개의 channel을 독립적으로 조절할 수
있고, 그 시점에 PySCF 에너지가 재현되었다.

### 물리적 의미와 적용 범위

확인된 계산 사실은 P4가 이 Gamma-point fixed-occupation 문제의 기준
에너지를 재현했다는 것이다. 이에 대한 물리적 해석은, 두 Si 원자에 놓인
동일한 성격의 local AO 사이에서 bonding/antibonding 조합을 만드는 자유도가
이 조건의 occupied subspace를 표현하는 데 지배적이었다는 것이다. 추가된
P5-P7의 서로 다른 local orbital 간 mixing은 에너지를 더 낮추지 않았다.

다만 이 결론은 primitive-cell Si, Gamma point, 현재 AO ordering 및 fixed
occupation에 한정된다. 비-Gamma k-point, 낮아진 대칭성, 변형된 구조,
더 큰 basis 또는 occupation 최적화에서는 서로 다른 orbital 간 mixing이나
복소 mixing 방향이 추가로 필요할 수 있다.

## Density matrix 측정과 Pauli grouping

### 측정 대상

Pauli grouping은 ansatz generator나 Kohn-Sham Hamiltonian 항이 아니라,
회로가 만든 raw density matrix $\rho_{ij}$를 복원하는 observable에 적용된다.
3-qubit 실수 대칭 density matrix에는 Identity를 제외하고 35개의 서로 다른
Pauli observable이 필요하다. Identity 성분은 알려진 상수이므로 별도로
측정하지 않는다.

현재 구현은 PennyLane의 RLF heuristic을 사용한 general-commuting grouping을
적용한다.

$$
35\text{ Pauli observables}
\longrightarrow
6\text{ general-commuting groups}
$$

### Group 구성

| Group | 크기 | Pauli strings |
|---:|---:|---|
| 1 | 7 | `IIZ`, `IZI`, `IZZ`, `ZII`, `ZIZ`, `ZZI`, `ZZZ` |
| 2 | 7 | `IIX`, `IXI`, `IXX`, `XII`, `XIX`, `XXI`, `XXX` |
| 3 | 6 | `IZX`, `ZZX`, `IXZ`, `ZXZ`, `IYY`, `ZYY` |
| 4 | 6 | `XYY`, `ZIX`, `ZXI`, `XZZ`, `YZY`, `YYZ` |
| 5 | 5 | `YXY`, `YYX`, `ZXX`, `XIZ`, `XZI` |
| 6 | 4 | `XZX`, `YIY`, `XXZ`, `YYI` |

Group 1은 모두 $Z$-type이므로 computational basis에서 직접 측정할 수 있다.
나머지 group은 $X$, $Y$, $Z$가 섞여 있지만 전체 operator로 서로 commute한다.
따라서 qubit-wise commuting일 필요는 없으며, group별 공통 eigenbasis로
simultaneous diagonalization한 뒤 한 번의 probability 측정으로 group 안의
모든 expectation value를 복원한다.

### 측정 circuit 수

Gamma point에는 k-point가 하나이고, density matrix를 만들기 위해 8개의 AO
basis state를 각각 입력 상태로 준비한다. 각 입력 상태마다 6개 group을
측정하므로 한 번의 전체 raw density-matrix 평가에 필요한 measurement
circuit 수는

$$
8\text{ basis states}\times6\text{ groups}=48
$$

이다. 각 circuit은 지정된 `shots`만큼 sampling하며, 측정된 computational-
basis probability 8개와 미리 계산한 eigenvalue table을 이용해 Pauli
expectation value와 $8\times8$ density matrix를 재구성한다.

### Parameter-shift gradient를 포함한 측정 비용

`au-fixed P4`에는 최적화할 회로 parameter가 4개 있다. Parameter-shift
gradient는 parameter 하나마다 양과 음 방향의 2개 shifted circuit를 요구한다.
따라서 energy를 위한 primal 측정까지 포함한 optimizer iteration당 circuit
수의 근사값은

$$
N_{\mathrm{circuit/itr}}
=
N_{\mathrm{basis}}N_{\mathrm{group}}
\left(1+2N_\theta\right)
$$

이다. 현재 조건에서는

$$
N_{\mathrm{circuit/itr}}
=
8\times6\times(1+2\times4)
=
\boxed{432}
$$

이다. 1000 shots를 사용하면 optimizer iteration당 sampling 수는 약

$$
432\times1000
=
\boxed{432{,}000\text{ shots}}
$$

이다.

5 iterations에는 본 최적화만으로 약 2160개 circuit가 필요하다. 최적화가
끝난 뒤 최종 density와 energy를 다시 평가하는 48개 circuit를 포함하면

$$
N_{\mathrm{circuit,total}}
\approx
432\times5+48
=
\boxed{2208}
$$

이며 총 sampling 수는 약 220만 shots이다.

## IonQ QPU dry-run 비용: GC와 QWC 비교

### 계산 조건

다음 조건에서 optimizer의 `itr 0` 한 번에 필요한 전체 QPU 비용을
IonQ `qpu.forte-1`의 dry-run으로 평가했다.

- 시스템: primitive-cell Si, Gamma point
- occupation: fixed occupation
- ansatz: `au-fixed P4`
- Pauli generators: `YZZ`, `YIZ`, `YZI`, `YII`
- optimizer iteration: `itr 0` 한 번
- gradient: 4개 parameter의 parameter-shift gradient 포함
- shots/circuit: 100, 1000, 10000
- grouping: general commuting(GC), qubit-wise commuting(QWC)
- 비용 단위: IonQ cost API가 반환한 USD 예상 비용

P4에는 $N_\theta=4$개의 parameter가 있으므로 각 AO basis/group batch에는
primal circuit 1개와 shifted circuit 8개가 포함된다.

$$
N_{\mathrm{circuit/job}}
=1+2N_\theta
=9
$$

### Grouping별 측정 규모

35개의 non-identity Pauli observable에 RLF grouping을 적용한 결과는 다음과
같다.

| Grouping | Group 수 | Batch job 수/itr | Circuit 수/job | 전체 circuit 수/itr |
|---|---:|---:|---:|---:|
| GC | 6 | $8\times6=48$ | 9 | 432 |
| QWC | 14 | $8\times14=112$ | 9 | 1008 |

GC는 서로 commute하지만 qubit-wise commute하지 않는 observable도 같은 group에
포함한다. QWC는 각 qubit에서 동일한 Pauli axis 또는 Identity만 허용하므로
measurement basis는 단순하지만 group 수가 14개로 증가한다.

### Shot 수와 dry-run 예상 비용

| Shots/circuit | GC 전체 shots | GC 예상 비용 | QWC 전체 shots | QWC 예상 비용 | QWC/GC |
|---:|---:|---:|---:|---:|---:|
| 100 | 43,200 | \$11,141.28 | 100,800 | \$25,996.32 | 2.33 |
| 1,000 | 432,000 | \$72,660.48 | 1,008,000 | \$169,541.12 | 2.33 |
| 10,000 | 4,320,000 | \$189,972.72 | 10,080,000 | \$411,185.52 | 2.16 |

위 비용은 각 AO basis/group 조합을 실제 dry-run batch job으로 제출한 뒤,
각 job의 `/jobs/{id}/cost`가 반환한 `estimated_cost`를 모두 합산한 값이다.
따라서 대표 circuit 하나의 비용을 단순 배수한 결과가 아니다.

### 확인된 사실

1. 현재 측정 구조에서 GC는 QWC보다 group 수와 circuit 수가 모두 작다.
2. 동일한 shot 조건에서 QWC 비용은 GC 비용의 약 2.2--2.3배다.
3. 비용은 shot 수에 단순 비례하지 않는다. IonQ 응답에 기록된 cost model은
   `2QGE_operations`이며, compiled circuit 구조와 shot 수가 함께 반영된다.
4. 1000 shots의 `itr 0` 한 번만으로도 GC 약 \$72,660, QWC 약 \$169,541의
   예상 비용이 반환됐다.

### 해석과 적용 범위

현재 시스템에서는 GC의 entangling measurement-basis rotation 비용보다 QWC의
group 수 증가로 인한 반복 측정 비용이 더 크다. 따라서 QPU 비용만 고려하면
QWC보다 GC가 유리하다.

이 결론은 현재 3-qubit structured AO encoding, P4 ansatz, `qpu.forte-1`의
dry-run cost model과 현재 계정 조건에 대한 결과다. IonQ 가격 정책, backend,
compiler 또는 계약 조건이 바뀌면 절대 비용은 달라질 수 있다. 다만 현재
회로에서 GC 6개와 QWC 14개라는 group 수 차이는 코드와 observable 구조에서
직접 확인된 값이다.

### 재현 명령과 원본 데이터

QWC grouping은 다음과 같이 선택한다.

```bash
python3 qudft.py Si --qpu --group --grouping qwc
```

비용 견적은 `ionq_dry_run_cost.py`로 생성했다.

```bash
python3 ionq_dry_run_cost.py \
  --shots 1000 \
  --grouping gc \
  --output data/ionq_cost_itr0_gc_s1000.json
```

원본 dry-run job ID와 job별 예상 비용은 다음 파일에 보존돼 있다.

```text
data/ionq_cost_itr0_gc_s100.json
data/ionq_cost_itr0_gc_s1000.json
data/ionq_cost_itr0_gc_s10000.json
data/ionq_cost_itr0_qwc_s100.json
data/ionq_cost_itr0_qwc_s1000.json
data/ionq_cost_itr0_qwc_s10000.json
```

### 원격 IonQ simulator의 시간 병목

현재 `measure_rho_pauli_groups`는 8개 basis state와 6개 group을 순차적으로
순회한다. PennyLane/IonQ backend가 한 QNode의 primal과 parameter-shift
circuit를 batch로 묶더라도 optimizer iteration당 최대 48회의 원격 QNode
batch 호출이 발생할 수 있다. 5 iterations와 최종 평가를 합치면 약 288회의
원격 batch 호출에 해당한다.

따라서 실제 소요 시간은 simulator의 순수 계산량보다 다음 요소에 크게
영향받는다.

- IonQ API job 제출 latency
- 원격 queue 대기 시간
- 각 job의 상태 polling 시간
- Basis state와 group을 순차 실행하는 현재 Python loop

현재 실행은 `fixed-occ`이므로 엄밀히 말하면 SCF iteration이 아니라 optimizer
iteration이다. Occupation은 고정되어 있지만 각 optimizer iteration에서
density matrix와 4개 parameter의 energy gradient를 전부 다시 측정한다.

1000-shot sampling noise의 대표적인 크기는 $O(1/\sqrt{1000})\approx0.03$이다.
Observable과 gradient의 실제 분산은 circuit마다 다르지만, 현재 optimization
tolerance $10^{-4}$는 이 sampling scale보다 훨씬 작다. 따라서 finite-shot
실행에서는 tolerance를 만족하지 못하고 `max_itr`까지 진행할 가능성을
고려해야 한다.

### 측정 비용 절감 방향

- Gradient 없이 energy와 density 측정 경로만 검증하면 평가당 48개 circuit로
  줄어든다.
- 기능 검증 단계에서는 shots와 `max_itr`을 작게 설정한다.
- 원격 실행 전에 finite-shot local simulator로 전체 경로를 검증한다.
- 모든 parameter를 매 iteration 측정하는 대신 부분 또는 순환 업데이트를
  사용할 수 있다.
- Basis-state/group별 QNode를 더 큰 원격 batch로 합치는 실행 구조를 검토할
  수 있다.

### 구현과 실행

- Group 생성: `lib/grouped_vqe.py`의 `_setup_rho_pauli_groups`
- RLF GC grouping: `grouping_type='commuting'`, `method='rlf'`
- RLF QWC grouping: `grouping_type='qwc'`, `method='rlf'`
- 공통 eigenbasis 구성: `_gc_diagonalization`
- QWC tensor-product basis 구성: `_qwc_diagonalization`
- Density matrix 복원: `measure_rho_pauli_groups`

실행 옵션은 다음과 같다.

```bash
python3 qudft.py Si --qpu --group --grouping gc
```

`--group`은 현재 `--qpu`와 함께 사용해야 한다. Grouping 결과와 simultaneous-
diagonalization unitary는 AO-to-basis mapping을 포함한 cache key로 NPZ 파일에
저장한다. `--grouping`에는 `gc`와 `qwc`를 지정할 수 있다.

## 원본 계산 데이터

대용량 원본은 복사하지 않았으며 repository의 다음 HDF5 파일에 있다.

아래 원본은 ansatz 명칭을 분리하기 전에 생성되어 파일명과 내부 metadata에
이전 명칭인 `au-adapt`가 남아 있다. 계산 규칙은 현재 `au-swap`과 같다.

```text
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P1_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P2_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P3_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P4_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P5_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P6_fixed-occ.out
data/Si_sc1x1x1_k1x1x1_qudft_au-adapt_P7_fixed-occ.out
```
