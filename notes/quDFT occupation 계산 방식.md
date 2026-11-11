---
created: 2026-11-10T13:48:09+09:00
updated: 2026-11-11T14:16:01+09:00
tags:
  - qudft
  - occupation
  - fractional-occupation
  - fock
  - scf
related:
  - "[[quDFT Ansatz 구성 규칙]]"
  - "[[Si Gamma-point primitive cell quDFT 분석]]"
---

# quDFT occupation 계산 방식

## 목차

- [[#공통 정의|공통 정의]]
- [[#Fixed occupation|Fixed occupation]]
	- [[#Fixed occupation#계산식|계산식]]
	- [[#Fixed occupation#최적화와 수렴|최적화와 수렴]]
- [[#Fock occupation|Fock occupation]]
	- [[#Fock occupation#Iteration 순서|Iteration 순서]]
	- [[#Fock occupation#순환 의존성의 처리|순환 의존성의 처리]]
	- [[#Fock occupation#최적화와 수렴|최적화와 수렴]]
- [[#Variational eta occupation|Variational eta occupation]]
	- [[#Variational eta occupation#계산식|계산식]]
	- [[#Variational eta occupation#최적화와 수렴|최적화와 수렴]]
- [[#온도와 free energy|온도와 free energy]]
- [[#PySCF reference occupation|PySCF reference occupation]]
- [[#방식 비교|방식 비교]]
- [[#제약과 적용 범위|제약과 적용 범위]]
- [[#구현 위치|구현 위치]]

## 공통 정의

각 k-point $k$와 circuit orbital $b$의 occupation을 $f_{kb}$로 표시한다.
현재 코드는 spin-restricted occupation을 사용하므로 범위는

$$
0 \le f_{kb} \le 2
$$

이다. AO basis의 density matrix는 circuit이 생성한 orbital projector
$P_{kb}(\theta)$를 occupation으로 가중하여 구성한다.

$$
\rho_k(\theta,f)
=
\sum_b f_{kb}P_{kb}(\theta)
$$

전자 수 제약은 모든 k-point에 동일한 가중치가 적용되는 현재 regular
k-grid에서 다음과 같다.

$$
\sum_{kb} f_{kb}
=
N_kN_e
$$

여기서 $N_k$는 k-point 수이고 $N_e$는 primitive cell의 전자 수이다.

## Fixed occupation

별도 occupation option을 지정하지 않았을 때 사용하는 기본 방식이다.
`--fixed-occ`로 명시할 수도 있다. Aufbau 순서의 occupation을 계산 시작 시
한 번 구성하고 모든 iteration에서 그대로 사용한다.

### 계산식

짝수 전자계에서는

$$
f_b=
\begin{cases}
2, & b < N_e/2,\\
0, & b \ge N_e/2
\end{cases}
$$

이다. 홀수 전자계에서는 마지막 singly occupied orbital에 $f_b=1$을 준다.
동일한 occupation 배열을 모든 k-point에 적용한다.

이 방식은 orbital energy를 기준으로 iteration마다 orbital을 다시 정렬하지
않는다. 따라서 고정되는 것은 초기 circuit-orbital index에 대응하는 Aufbau
occupation pattern이다.

### 최적화와 수렴

최적화 변수는 $\theta$뿐이다. $\eta$는 빈 배열로 전달되며 optimizer
parameter, gradient, 결과 dataset에서 제거된다.

$$
x=\theta,
\qquad
N_\eta=0
$$

수렴 조건에서는 `grad_eta`를 평가하지 않고 내부적으로 $0$으로 취급한다.
따라서 실질적인 조건은

$$
\|\nabla_\theta F\|_{\mathrm{RMS}} < \epsilon_g,
\qquad
|F_i-F_{i-1}| < \epsilon_F
$$

이다.

## Fock occupation

`--fock-occ`로 선택한다. $\eta$를 최적화하지 않고, 각 SCF iteration에서
circuit orbital의 Fock expectation value로 fractional occupation을 다시
계산한다.

### Iteration 순서

Iteration $i$에서 사용하는 순서는 다음과 같다.

$$
F^{(i)}
\longrightarrow
\varepsilon_{kb}^{(i)}
=
\operatorname{Tr}
\left[F_k^{(i)}P_{kb}(\theta^{(i)})\right]
$$

$$
\varepsilon^{(i)}
\longrightarrow
f^{(i)}
\longrightarrow
\rho^{(i)}
\longrightarrow
F^{(i+1)}
$$

$f^{(i)}$는 `mf.get_occ`가 Fermi–Dirac 분포와 정확한 전자 수 제약을
사용하여 계산한다. $\sigma=0$이면 energy가 낮은 순서대로 $2$, 필요하면
$1$, 그 뒤에 $0$을 배치한다.

첫 iteration의 $F^{(0)}$는 PySCF initial density에서 만든다.

### 순환 의존성의 처리

Occupation을 구하려면 density에서 계산한 Fock matrix가 필요하고, density를
구하려면 occupation이 필요한 순환 구조가 있다. 현재 구현은 같은
iteration 안에서 이를 동시에 푸는 대신 이전 단계의 Fock matrix를
사용하는 lagged SCF update로 해결한다.

즉, $F^{(i)}$로 $f^{(i)}$를 먼저 정한 뒤 $\rho^{(i)}$와 $F^{(i+1)}$를
계산한다. Occupation은 automatic differentiation 밖에서 산출하며,
$\theta$ gradient를 계산할 때는 $f^{(i)}$를 상수로 취급한다.

따라서 Fock matrix를 별도의 변분 parameter로 목적함수에 추가하지 않는다.
대신 Fock–density self-consistency를 외부 iteration으로 달성한다.

### 최적화와 수렴

변분 optimizer가 직접 갱신하는 변수는 $\theta$뿐이다.

$$
x=\theta
$$

그러나 occupation과 Fock matrix는 optimizer 바깥의 SCF state로 매
iteration 갱신된다. 수렴 조건은 다음 세 조건이다.

$$
\|\nabla_\theta F\|_{\mathrm{RMS}} < \epsilon_g
$$

$$
\max_{kb}|f_{kb}^{(i)}-f_{kb}^{(i-1)}| < \epsilon_F
$$

$$
|F_i-F_{i-1}| < \epsilon_F
$$

결과에는 `eta=[]`, `grad_eta=0.0`을 저장하고, 대신 `occ_history`,
`eks_history`, 최종 `fock`을 저장한다.

## Variational eta occupation

`--var-occ`로 선택한다. Circuit parameter $\theta$와 occupation parameter
$\eta$를 하나의 변분 문제에서 동시에 최적화한다.

### 계산식

Occupation은 sigmoid로 parameterization한다.

$$
f_{kb}
=
2\sigma(\eta_{kb}+\mu)
=
\frac{2}{1+\exp[-(\eta_{kb}+\mu)]}
$$

공통 shift $\mu$는 전자 수 제약

$$
2\sum_{kb}\sigma(\eta_{kb}+\mu)
-N_kN_e
=0
$$

을 만족하도록 20회의 Newton–Raphson iteration으로 구한다. 따라서 $\eta$가
변하더라도 전체 전자 수는 유지된다.

초기 $\eta$는 paired occupied orbital에 `4.0`, unoccupied orbital에
`-4.0`을 주며, 홀수 전자가 있으면 경계 orbital에 `0.0`을 준다.

### 최적화와 수렴

최적화 변수는 다음과 같다.

$$
x=(\theta,\eta)
$$

$\theta$와 $\eta$의 gradient를 모두 계산한다. 두 parameter 군은 같은
optimizer state를 사용하지만 $\eta$ update에는
`lr_eta / lr_theta`가 추가로 곱해진다.

일반 backend의 수렴 조건은 다음 세 조건을 모두 만족하는 것이다.

$$
\|\nabla_\theta F\|_{\mathrm{RMS}} < \epsilon_g
$$

$$
\|\nabla_\eta F\|_{\mathrm{RMS}} < \epsilon_g
$$

$$
|F_i-F_{i-1}| < \epsilon_F
$$

이 방식은 occupation까지 변분 자유도로 포함하므로 orbital rotation만
최적화하는 방식보다 수렴이 느릴 수 있다.

## 온도와 free energy

기본 목적함수는 finite-temperature free energy이다.

$$
F=E_{\mathrm{tot}}-\sigma S
$$

$$
S
=
-\frac{2}{N_k}
\sum_{kb}
\left[
p_{kb}\ln p_{kb}
+(1-p_{kb})\ln(1-p_{kb})
\right],
\qquad
p_{kb}=\frac{f_{kb}}{2}
$$

`--ground-state`를 사용하면 occupation 방식은 유지하면서 목적함수만

$$
F=E_{\mathrm{tot}}
$$

로 바뀐다. 따라서 `--ground-state`는 fixed occupation을 의미하지 않는다.

짝수 전자계의 `--fixed-occ`처럼 모든 occupation이 $0$ 또는 $2$이면
$S=0$이므로 `efree=etot`이다. 홀수 전자계에서 $f=1$이 포함되면 fixed
occupation이어도 finite-temperature entropy가 $0$이 아닐 수 있다.

## PySCF reference occupation

`--pyscf` reference 계산과 `--fock-occ`는 `mf.get_occ`를 프로젝트의
fractional occupation 함수로 교체한다. Orbital energy를
$\varepsilon_i$라 하면 $\sigma>0$에서

$$
f_i
=
\frac{2}{1+\exp[(\varepsilon_i-\mu)/\sigma]}
$$

를 사용한다. $\mu$는 bisection으로 전체 occupation 합이 정확한 전자 수와
같아지도록 결정한다. 이 구현은 홀수 전자 수도 보존한다.

## 방식 비교

| 방식 | 선택 방법 | 직접 최적화 변수 | Occupation 갱신 | 전자 수 보존 | 주요 수렴 지표 |
|---|---|---|---|---|---|
| Fixed | 기본값 또는 `--fixed-occ` | $\theta$ | 갱신하지 않음 | 고정 Aufbau 배열 | `grad_theta`, `delta_efree` |
| Fock | `--fock-occ` | $\theta$ | 매 SCF iteration | Fermi $\mu$ | `grad_theta`, `delta_occ`, `delta_efree` |
| Variational eta | `--var-occ` | $\theta,\eta$ | 매 optimizer step | 공통 $\mu$ | `grad_theta`, `grad_eta`, `delta_efree` |

## 제약과 적용 범위

- `--fixed-occ`, `--fock-occ`, `--var-occ`는 상호 배타적이다.
- 현재 `--fock-occ`는 `au-swap`, `au-grow`와 함께 사용할 수 없다.
- 현재 `--fock-occ`는 resume을 지원하지 않는다.
- `--fixed-occ`는 band crossing이나 k-point별 occupation 재배치를 반영하지
  않으므로 Gamma-point 절연체와 같이 occupation pattern이 명확한 경우에
  가장 직접적이다.
- `--fock-occ`는 fractional occupation과 orbital energy ordering을
  self-consistent하게 갱신해야 하는 k-grid 계산에 더 자연스럽다.
- `ionq.qpu`에서는 gradient tolerance를 $10^{-2}$, free-energy 및
  occupation tolerance를 $10^{-3}$으로 완화한다. `ionq.simulator`는 CLI
  기본 `--tol` 값인 $10^{-4}$를 두 tolerance에 모두 사용한다.

## 구현 위치

- CLI 선택 및 초기화: `qudft.py`
- Fractional Fermi occupation: `lib/occupations.py`
- eta occupation과 전자 수 제약: `lib/vqe.py`의 `get_occ`
- Fixed occupation 선택: `lib/vqe.py`의 `optimization_occ`
- Fock expectation occupation: `lib/vqe.py`의 `get_fock_occupation`
- 목적함수: `lib/vqe.py`의 `cost_qudft`, `cost_qudft_occ`
- 최적화 loop: `lib/vqe.py`의 `qudftv2`, `qudftv2_fock_occ`
