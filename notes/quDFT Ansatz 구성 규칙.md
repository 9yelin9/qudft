---
created: 2026-11-06T16:03:12+09:00
updated: 2026-11-10T14:01:36+09:00
tags:
  - qudft
  - ansatz
  - pauli
  - orbital-rotation
related:
  - "[[Structured AO binary encoding]]"
  - "[[Si Gamma-point primitive cell quDFT 분석]]"
  - "[[quDFT occupation 계산 방식]]"
---

# quDFT Ansatz 구성 규칙

## 목차

- [[#공통 구조|공통 구조]]
- [[#`au-oddy`|`au-oddy`]]
	- [[#`au-oddy`#au-oddy 구현 규칙|au-oddy 구현 규칙]]
- [[#`au-all`|`au-all`]]
	- [[#`au-all`#au-all 구현 규칙|au-all 구현 규칙]]
- [[#`au-prio`|`au-prio`]]
	- [[#`au-prio`#후보군|후보군]]
	- [[#`au-prio`#Greedy priority|Greedy priority]]
- [[#Adaptive Pauli pool 공통 규칙|Adaptive Pauli pool 공통 규칙]]
	- [[#Adaptive Pauli pool 공통 규칙#전체 pool|전체 pool]]
	- [[#Adaptive Pauli pool 공통 규칙#초기 active 선택|초기 active 선택]]
- [[#`au-swap`|`au-swap`]]
	- [[#`au-swap`#Active 크기|Active 크기]]
	- [[#`au-swap`#Swap 검토|Swap 검토]]
- [[#`au-grow`|`au-grow`]]
	- [[#`au-grow`#Active 크기 증가|Active 크기 증가]]
	- [[#`au-grow`#현재 회로 표현 방식|현재 회로 표현 방식]]
	- [[#`au-grow`#저장 데이터|저장 데이터]]
	- [[#`au-grow`#구현 검증|구현 검증]]
- [[#`ao-full`|`ao-full`]]
	- [[#`ao-full`#ao-full 구현 규칙|ao-full 구현 규칙]]
- [[#`rycn`|`rycn`]]
	- [[#`rycn`#rycn 구현 규칙|rycn 구현 규칙]]
- [[#Occupation 옵션과 ansatz의 관계|Occupation 옵션과 ansatz의 관계]]
- [[#구현 위치|구현 위치]]

## 공통 구조

Structured AO encoding은 computational basis를 atom index와 local AO index로
분리한다.

$$
\mathcal H
=
\mathcal H_{\mathrm{atom}}
\otimes
\mathcal H_{\mathrm{local\ AO}}
$$

$n$-qubit Pauli rotation은 다음과 같다.

$$
U_P(\theta)=\exp\left(-\frac{i\theta P}{2}\right)
$$

Pauli string에서 `X` 또는 `Y`가 있는 wire는 computational-basis bit를
뒤집는다. 따라서 string $P$의 orbital-pair coverage는 해당 flip mask로
연결되는 유효 AO basis-state pair의 집합이다. `I`와 `Z`는 bit를 뒤집지
않지만 연결된 pair의 부호 패턴을 바꿀 수 있다.

## `au-oddy`

### au-oddy 구현 규칙

- Identity를 제외한 Pauli group에서 `Y`의 개수가 홀수인 string만 사용한다.
- 모든 선택된 string에 독립적인 `PauliRot` 파라미터를 부여한다.
- 별도의 priority 또는 truncation은 적용하지 않는다.

Odd-Y string의 수는 다음과 같다.

$$
N_{\mathrm{odd-Y}}
=
\frac{4^n-2^n}{2}
$$

3-qubit Si에서는 $N_{\mathrm{odd-Y}}=28$이다. Odd-Y 조건은
$-iP$가 실수 generator가 되어 real orbital rotation을 생성하도록 한다.

## `au-all`

### au-all 구현 규칙

- $\{I,X,Y,Z\}^{\otimes n}$의 모든 string 중 Identity만 제외한다.
- pool과 실제 회로가 동일하며 모든 string에 독립 파라미터를 부여한다.

$$
N_{\mathrm{au-all}}=4^n-1
$$

3-qubit에서는 63개 `PauliRot`을 사용한다. `--gradient-history`를 사용하면
각 iteration의 63개 parameter gradient와 Pauli label을 HDF5에 저장할 수
있다.

## `au-prio`

### 후보군

`au-prio`의 후보군은 `au-oddy`와 동일한 odd-Y Pauli string이다.

### Greedy priority

선택된 집합을 $\mathcal S$라고 할 때 후보 $P$는 다음 tuple의 오름차순으로
선정된다.

$$
\operatorname{key}(P)
=
\left(
-C_{\mathrm{new}}(P),
-N_{\mathrm{noncommute}}(P),
W(P),
I_{\mathrm{inter}}(P),
L(P)
\right)
$$

- $C_{\mathrm{new}}$: 아직 덮이지 않은 유효 AO pair-direction 수
- $N_{\mathrm{noncommute}}$: 기존 선택 string과 반교환하는 횟수
- $W$: Pauli weight
- $I_{\mathrm{inter}}$: inter이면 1, intra이면 0이므로 tie에서 intra 우선
- $L$: 문자열 사전식 deterministic tie-break

두 Pauli string은 서로 다른 non-identity Pauli가 겹치는 wire의 수가 홀수면
반교환한다. 현재 후보가 모두 odd-Y이므로 direction label은 모두 real
rotation 방향 `R`이다.

`--nparam P`는 priority prefix의 처음 $P$개를 사용한다.
`--min-coverage`는 이 prefix가 모든 유효 AO pair-direction을 덮는지 검사한다.

## Adaptive Pauli pool 공통 규칙

### 전체 pool

- 전체 후보 pool은 Identity를 제외한 모든 Pauli string이다.
- 3-qubit에서는 pool 크기가 63이다.
- 현재 구현은 full AO-pair coverage를 강제하지 않는다.

### 초기 active 선택

모든 Pauli parameter가 0인 상태에서 63개 gradient를 계산한다. k-grid에서는
같은 Pauli의 k-point별 gradient를 다음 norm으로 합친다.

$$
s_j
=
\sqrt{
\sum_k
\left|
\frac{\partial E}{\partial\theta_{kj}}
\right|^2
}
$$

$s_j$가 큰 순서로 초기 active string을 선택한다. 모든 k-point는 같은
active Pauli 구조를 공유하지만 $\theta_{kj}$는 k-point마다 독립적이다.

## `au-swap`

### Active 크기

- `--nparam P`는 실행 내내 유지되는 active Pauli 수 $P$이다.
- 초기에는 gradient score가 가장 큰 $P$개를 선택한다.
- 구조 변경은 active string 하나와 excluded string 하나를 바꾸는 1-to-1
  swap만 허용한다.

### Swap 검토

Active gradient는 다음 L2 norm이다.

$$
G_{\mathrm{active}}
=
\sqrt{
\sum_{k}
\sum_{j\in\mathcal A}
\left|
\frac{\partial E}{\partial\theta_{kj}}
\right|^2
}
$$

현재 상수는 다음과 같다.

```text
active_tol      = 1e-4
excluded_tol    = 5e-4
patience        = 10
trial_steps     = 20
min_improvement = 1e-6 Ha
```

$G_{\mathrm{active}}<\texttt{active\_tol}$이 `patience`회 연속 유지될 때만
excluded insertion gradient를 검사한다. 현재 회로 뒤에 각 excluded
generator를 0 parameter로 삽입한 gradient의 k-point 합성 norm 중 최댓값을
사용한다.

$$
G_{\mathrm{excluded}}
=
\max_{P\notin\mathcal A}
\sqrt{
\sum_k
\left|
\left.
\frac{\partial E}{\partial\phi_{kP}}
\right|_{\phi_{kP}=0}
\right|^2
}
$$

$G_{\mathrm{excluded}}>\texttt{excluded\_tol}$이면 가장 큰 후보를 선택한다.
기존 active string을 하나씩 제거해 만든 모든 1-to-1 trial 회로를 각각
`trial_steps`회 최적화한다. 가장 낮은 trial energy가 현재 energy보다 다음
값 이상 낮을 때만 swap을 승인한다.

$$
E_{\mathrm{trial}}
<
E_{\mathrm{current}}
-10^{-6}\ E_h
$$

승인 시 유지된 parameter는 전달하고 새 string의 parameter는 0에서
시작하며 optimizer state를 초기화한다.

## `au-grow`

### Active 크기 증가

- 초기 active set은 gradient score가 가장 큰 Pauli string 하나로 시작한다.
- `--nparam P`는 초기 크기가 아니라 최대 active Pauli 수 $P$이다.
- $G_{\mathrm{active}}<10^{-4}$가 10회 연속 유지되면 excluded insertion
  gradient를 평가한다.
- $G_{\mathrm{excluded}}>5\times10^{-4}$이면 가장 큰 excluded string을
  active set 뒤에 추가한다.
- 기존 string을 제거하거나 교체하지 않으며 active 수가 한 개씩 증가한다.
- 추가된 parameter는 0에서 시작하고 기존 active parameter는 유지한다.
- 최대 active 수에 도달하거나 유효한 excluded 방향이 남지 않으면 더 이상
  회로를 늘리지 않는다.

즉, 두 adaptive ansatz의 구조 변화는 다음과 같이 구분된다.

| Ansatz | `--nparam` 의미 | 초기 active 수 | 구조 변경 | Active 수 |
|---|---|---:|---|---|
| `au-swap` | 고정 active 수 | $P$ | 1-to-1 교체 | 항상 $P$ |
| `au-grow` | 최대 active 수 | 1 | generator 추가 | 1부터 최대 $P$ |

### 현재 회로 표현 방식

현재 QNode에는 pool 전체의 `PauliRot`이 들어가고 inactive parameter를 0으로
mask한다. 따라서 수학적으로 유효한 active gate는 $P$개지만 QNode에는
$4^n-1$개의 rotation operation이 존재한다. Active set이 바뀌면 active
string을 앞쪽으로 재정렬하고 QNode를 다시 생성한다.

### 저장 데이터

두 adaptive ansatz 결과에는 다음 공통 history가 포함된다.

```text
gradient_paulis
grad_theta_history
active_index_history
efree_history
grad_eta_history
occ_history
swap_iteration
swap_energy_change
grow_iteration
grow_pauli
```

`au-swap`에서는 `swap_*`에 기록되고 `grow_*`는 비어 있다. `au-grow`에서는
반대로 `grow_*`에 추가 iteration과 Pauli label이 기록된다.

### 구현 검증

Primitive-cell Si Gamma-point fixed-occupation 단기 실행에서 확인한 초기
active set은 다음과 같다.

- `au-swap --nparam 4`: `YZZ`, `YIZ`, `YZI`, `YII`
- `au-grow --nparam 4`: `YZZ` 한 개

`au-grow`를 300 iteration까지 실행한 검증에서는 iteration 227에
$G_{\mathrm{excluded}}\approx6\times10^{-6}$으로 평가되어
`excluded_tol=5\times10^{-4}`보다 작았다. 따라서 이 실행에서는 generator가
추가되지 않고 active 수 1에서 수렴했다. 이는 grow 기능의 오류가 아니라
현재 insertion-gradient threshold 규칙에 따른 결과이며, P1의 잘못된
stationary point에서 gradient만으로는 성장 필요성을 감지하지 못할 수 있음을
보여준다.

## `ao-full`

### ao-full 구현 규칙

- Gamma point에서만 사용할 수 있다.
- 모든 원자 내부 AO pair를 먼저 나열한다.
- 이후 서로 다른 원자에 속하는 모든 AO pair를 나열한다.
- 각 pair에 독립적인 exact two-level real $R_Y$ mixing을 적용한다.

원자 내부와 원자 간 generator는 각각 다음 구조를 갖는다.

$$
K_{\mathrm{intra}}
=
\begin{pmatrix}
K_{AA} & 0\\
0 & K_{BB}
\end{pmatrix}
$$

$$
K_{\mathrm{inter}}
=
\begin{pmatrix}
0 & K_{AB}\\
K_{AB}^{\dagger} & 0
\end{pmatrix}
$$

현재 circuit 적용 순서는 intra mixing 이후 inter mixing이다.

## `rycn`

### rycn 구현 규칙

- $P$개의 `RY`를 wire 순환 순서로 배치한다.
- $C$개의 `CNOT`을 전체 `RY` slot에 가능한 한 균등하게 분배한다.
- topology는 `linear`, `reverse`, `ring`, `all` 중 하나다.
- `--circuit FILE`을 주면 JSON의 `ry_wires`와 `cnot_after`를 사용하며 CLI의
  topology, `nparam`, `ncnot`보다 우선한다.

Topology별 CNOT pair 규칙은 다음과 같다.

- `linear`: 인접 wire의 정방향 연결
- `reverse`: 인접 wire의 역방향 연결
- `ring`: `linear`에 마지막 wire에서 첫 wire로의 연결 추가
- `all`: 모든 서로 다른 wire pair의 양방향 연결

## Occupation 옵션과 ansatz의 관계

Occupation 선택은 ansatz 종류와 독립적이다.

- 기본값 또는 `--fixed-occ`: Aufbau occupation을 고정하고 eta를 제거
- `--fock-occ`: circuit orbital의 Fock expectation value로 occupation 갱신
- `--var-occ`: eta를 통해 fractional occupation을 variational optimization
- `--ground-state`: 선택된 occupation 방식은 유지하고 목적함수를
  zero-temperature `etot`으로 변경

Fixed occupation Gamma-point Si의 최소 active Pauli 결과는
[[Si Gamma-point primitive cell quDFT 분석]]에 정리되어 있다.

## 구현 위치

```text
lib/ansatz.py
lib/vqe.py
qudft.py
```
