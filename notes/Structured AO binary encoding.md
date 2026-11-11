---
created: 2026-11-06T16:47:21+09:00
updated: 2026-11-10T13:46:36+09:00
tags:
  - qudft
  - ao-encoding
  - binary-encoding
  - pennylane
related:
  - "[[quDFT Ansatz 구성 규칙]]"
  - "[[Si Gamma-point primitive cell quDFT 분석]]"
---

# Structured AO binary encoding

## 목차

- [[#핵심 원리|핵심 원리]]
- [[#PennyLane wire 순서|PennyLane wire 순서]]
- [[#Primitive-cell Si 예시|Primitive-cell Si 예시]]
- [[#Mixing의 구조적 의미|Mixing의 구조적 의미]]
- [[#일반적인 원자 수와 AO 수|일반적인 원자 수와 AO 수]]
- [[#확인된 구현 위치|확인된 구현 위치]]
- [[#적용 범위|적용 범위]]

## 핵심 원리

Structured AO binary encoding은 AO의 연속 번호를 그대로 binary string으로
바꾸는 대신, computational basis를 atom index와 원자 내부의 local AO
index로 분리한다.

$$
\mathcal H
=
\mathcal H_{\mathrm{atom}}
\otimes
\mathcal H_{\mathrm{local\ AO}}
$$

AO를 atom $a$와 local AO $\ell$의 순서쌍으로 표시하면 encoding index는

$$
j(a,\ell)=a,2^{n_{\mathrm{local}}}+\ell
$$

이다. 필요한 bit 수는

$$
n_{\mathrm{atom}}
=
\left\lceil\log_2 N_{\mathrm{atom}}\right\rceil,
\qquad
n_{\mathrm{local}}
=
\left\lceil\log_2 L_{\max}\right\rceil
$$

이며 $L_{\max}$는 한 원자가 가진 AO 수의 최댓값이다. 전체 qubit 수는

$$
n_{mathrm{qubit}}=n_{\mathrm{atom}}+n_{\mathrm{local}}
$$

이다. 단, 구현에서는 전체 AO 공간이 1차원인 특수한 경우에도 PennyLane
device를 위해 한 wire를 둔다.

## PennyLane wire 순서

현재 구현은 PennyLane의 `BasisState` 순서를 그대로 따른다. 첫 번째 배열
원소인 wire 0이 computational-basis index의 most-significant bit이다.
따라서 wire는 다음 순서로 배치된다.

$$
|q_0q_1\cdots q_{n-1}\rangle
=
|\underbrace{\text{atom bits}}_{\text{낮은 wire 번호}}
\underbrace{\text{local AO bits}}_{\text{높은 wire 번호}}\rangle
$$

수학 교재에서 흔히 쓰는 $q_{n-1}\cdots q_0$ 표기와 혼동하지 않도록,
이 프로젝트에서는 문자열의 왼쪽 문자가 항상 PennyLane wire 0에 대응한다.

## Primitive-cell Si 예시

두 Si 원자 $A,B$가 각각 4개의 local AO를 가지면
$n_{\mathrm{atom}}=1$, $n_{\mathrm{local}}=2$이고 wire의 역할은 다음과
같다.

| PennyLane wire | 역할 |
|---:|---|
| 0 | atom index |
| 1 | local AO index의 상위 bit |
| 2 | local AO index의 하위 bit |

AO mapping은 다음과 같다.

| Binary state | Atom | Local AO |
|---|---|---:|
| `000` | A | 0 |
| `001` | A | 1 |
| `010` | A | 2 |
| `011` | A | 3 |
| `100` | B | 0 |
| `101` | B | 1 |
| `110` | B | 2 |
| `111` | B | 3 |

Local AO 순서가 $s,p_x,p_y,p_z$라면 `000`과 `100`은 서로 다른 원자의
$s$ orbital, `001`과 `101`은 서로 다른 원자의 $p_x$ orbital에 대응하는
식이다. 실제 orbital label은 PySCF AO ordering에 의해 정해지므로 basis가
바뀌면 이를 별도로 확인해야 한다.

## Mixing의 구조적 의미

Pauli string의 `X`와 `Y`는 해당 bit를 뒤집고 `I`와 `Z`는 bit를 보존한다.
따라서 다음과 같이 해석할 수 있다.

- Atom wire만 뒤집으면 서로 다른 원자의 같은 local AO를 연결한다.
- Local-AO wire만 뒤집으면 같은 원자 내부의 서로 다른 AO를 연결한다.
- Atom wire와 local-AO wire를 함께 뒤집으면 서로 다른 원자의 서로 다른
  local AO를 연결한다.
- `Z`는 AO pair를 바꾸지 않지만 local channel별 부호와 mixing amplitude를
  구분하는 데 사용된다.

Primitive-cell Si의 최소 P4 set

$$
YII,\quad YIZ,\quad YZI,\quad YZZ
$$

은 atom wire만 뒤집으므로 모두
$|A,\ell\rangle\leftrightarrow|B,\ell\rangle$를 생성한다. 이 encoding 덕분에
해당 결과를 서로 다른 원자의 같은 orbital끼리의 mixing이 중요했다는
형태로 직접 해석할 수 있다.

## 일반적인 원자 수와 AO 수

원자 수나 원자별 AO 수가 2의 거듭제곱일 필요는 없다. 예를 들어 원자가
3개이면 두 atom bit를 사용하고 atom bit pattern 하나는 비어 있다. 원자별
AO 수가 다르면 적은 AO를 가진 원자의 일부 local state도 비어 있게 된다.

유효 AO 집합을 $\mathcal V$라고 하면 실제 계산과 pair coverage는 전체
$2^{n_{\mathrm{qubit}}}$ basis가 아니라

$$
\mathcal V
=
\{j(a,\ell)\mid 0\le a<N_{\mathrm{atom}},\ 0\le\ell<L_a\}
$$

에 대해서만 정의해야 한다. 현재 구현은 `ao_to_basis`에 이 유효 index만
저장하고, ansatz의 coverage 계산에도 같은 목록을 전달한다.

## 확인된 구현 위치

- `lib/ao_encoding.py`의 `structured_ao_encoding`: PySCF AO를 structured
  basis index로 변환한다.
- `VQE.atom_wires`: atom index를 담당하는 낮은 번호의 PennyLane wire이다.
- `VQE.local_wires`: local AO index를 담당하는 나머지 wire이다.
- `lib/ao_encoding.py`의 `encoding_index`: $(a,\ell)$을
  $a2^{n_{\mathrm{local}}}+\ell$로 변환한다.
- `intra_mixing`과 `inter_mixing`: 같은 encoding index를 사용해 정확한 두
  basis state 사이의 two-level $R_Y$ rotation을 만든다.

## 적용 범위

이 encoding은 Hilbert-space dimension을 줄이는 compression 기법이 아니라,
같은 binary dimension 안에서 AO에 물리적인 주소 체계를 부여하는 방법이다.
따라서 표현 가능한 상태 수는 일반 binary encoding과 같지만, atom 내부와
atom 사이의 mixing을 회로 구조 및 Pauli string으로 구분할 수 있다는 장점이
있다.
