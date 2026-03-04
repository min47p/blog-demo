---
title: "Generative Models - 05. Autoencoders, GAN"
date: 2026-02-21
tags: ["machine-learning", "generative-models"]
draft: false
---

# Latent Variable Models with Deterministic Decoder

이전 포스트에서 VAE에 대해 알아보면서 자연스러운 의문이 들었을 수 있다. VAE는 데이터 $\mathbf{x}$와 잠재 변수 $\mathbf{z}$ 사이의 관계를 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$라는 확률 분포로 모델링하고 있다. 그런데 확률 분포 대신 결정론적인 함수 $f_{\theta}(\mathbf{z})$를 정의해서 $\mathbf{x} = f_{\theta}(\mathbf{z})$가 되도록 하면 안 될까?

이렇게 결정론적인 함수를 이용해 생성 모델을 학습시키는 방법을 알아보자. 이번 포스트에서는 autoencoder와 generative adversarial network (GAN)에 대해 간단히 알아볼 것이다. 두 모델 사이에 큰 연관성은 없지만, 둘 모두 우리가 마지막에 살펴볼 diffusion model과 관련이 적으므로 한 포스트에 묶었다. 다음 포스트에서는 normalizing flow를 살펴볼 것이다.

$\mathbf{z}$의 공간인 latent space에서 $\mathbf{x}$의 공간인 data space로 가는 함수 $f_{\theta}$, 또는 $\mathbf{z}$가 주어진 $\mathbf{x}$의 조건부 확률 분포 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$ 등을 **decoder** 또는 **generator**라고 부른다. 반대로, data space에서 latent space로 가는 함수나 확률 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$ 등을 **encoder**라고 부른다. Encoder/decoder라는 용어는 VAE를 다룬 이전 포스트에서도 소개했다. VAE의 $p_{\theta}$와 $q_{\phi}$도 각각 decoder와 encoder이다. 하지만 VAE의 배경으로 variational Bayes를 더 강조하고 싶어서 이 용어들을 최대한 사용하지 않았다 (사실 autoencoder도 VAE의 중요한 배경임에도 불구하고). 앞으로는 이 용어들을 자유롭게 사용할 것이다. 그리고 GAN을 설명할 때는 decoder 대신 generator라는 단어를 사용할 것이다.

앞으로는 잠재 변수 $\mathbf{z}$의 prior로 매개변수 $\theta$에 의존하지 않는 고정된 분포(표준 정규 분포 등)를 사용하겠다. 따라서 $p_{\theta}(\mathbf{z})$ 대신 $p(\mathbf{z})$로 표기하겠다.

## Change of Variables

VAE에서와 같이 decoder가 조건부 확률 분포일 경우 (probabilistic decoder), 우리가 $p_{\mathrm{data}}$와 최대한 비슷하게 만들어야 하는 marginal distribution $p_{\theta}(\mathbf{x})$는 다음과 같다.
$$p_{\theta}(\mathbf{x}) = \int p_{\theta}(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$$

그럼 decoder가 결정론적이면 어떻게 될까? 얼핏 생각하면 $\mathbf{x}$가 주어질 때 $\mathbf{x} = f_{\theta}(\mathbf{z})$를 만족하는 $\mathbf{z}$를 찾아서 $p(\mathbf{x}) = p(\mathbf{z})$로 정의하면 될 것 같다. 만약 $f_{\theta}$가 단사함수가 아니라서 조건을 만족하는 $\mathbf{z}$가 여러 개라면 해당하는 모든 값에 대해 $p(\mathbf{z})$를 더하고, 영역으로 나타난다면 이 영역에 대해 $p(\mathbf{z})$를 적분하고... 이러한 접근은 잘못되었다. $\mathbf{x}$와 $\mathbf{z}$가 연속 확률 변수이기 때문이다.

{{< toggle title="Dirac delta function을 이용한 잘못된 접근" >}}
$p_{\theta}(\mathbf{x} \mid \mathbf{z})$는 Dirac delta function을 활용해 $p(\mathbf{x} \mid \mathbf{z}) = \delta(\mathbf{x} - f_{\theta}(\mathbf{z}))$로 나타낼 수 있으므로,
{{< eqlabel wrong-trial-with-delta >}}
$$p_{\theta}(\mathbf{x}) = \int \delta(\mathbf{x} - f_{\theta}(\mathbf{z}))\, p(\mathbf{z})\, d\mathbf{z} = \int_{f_{\theta}^{-1}(\mathbf{x})} \delta(\mathbf{0})\, p(\mathbf{z})\, d\mathbf{z}\, ? $$

뭔가 이상한 것 같다. 실제로 위 식은 두 가지 이유로 잘못된 식이다. 먼저, Dirac delta function은 일반적인 함수가 아니므로 $\delta(\mathbf{0})$과 같은 표현은 쓸 수 없다. 다음으로, 위 식에서는 $f^{-1}_{\theta}(\mathbf{x})$라는 식을 '$f_{\theta}(\mathbf{z}) = \mathbf{x}$를 만족하는 $\mathbf{z}$의 집합' 이라는 의미로 사용하고 있다. 그런데 이 집합의 (Lesbesgue) 측도는 거의 항상 $0$이다. 예를 들어, $f_{\theta}$가 단사함수인 경우 이 집합은 점 하나 또는 공집합이고, 이들의 측도는 모두 $0$이다. 그리고 집합의 측도가 $0$이라면 이 집합 위에서의 적분은 항상 $0$이 된다. $\delta(\mathbf{0}) = \infty$ 이므로 이 식이 값을 가질 수 있다고 생각할 수 있지만, 이 식은 처음부터 틀린 식이다.
{{< /toggle >}}

그렇다면 $p_{\theta}(\mathbf{x})$를 어떻게 다루어야 할까? 지금부터는 $\mathbf{z}$의 공간인 latent space와 $\mathbf{x}$의 공간인 data space의 차원에 신경써야 한다. 이들을 각각 $d$와 $D$라는 문자로 나타내자. 즉 $\mathbf{z} \in \mathbb{R}^{d}$, $\mathbf{x} \in \mathbb{R}^{D}$이고, $f_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{D}$이다. 이제 경우를 나누어 생각한다. 편의상 $f_{\theta}$의 치역을 $S$라 정의하자.

### Case 1: $d = D$이고, decoder의 역함수가 존재

첫 번째 케이스는 $d = D$이고 $f_{\theta}: \mathbb{R}^{D} \rightarrow \mathbb{R}^{D}$가 단사함수인 경우이다. 이때, $f_{\theta}$에는 역함수 $f^{-1}_{\theta}: S \rightarrow \mathbb{R}^{D}$가 존재한다. 이때, 확률 변수 간의 **change of variables formula**를 활용할 수 있다.

{{< callout type="Change of Variables">}}
$\mathbf{z} \in \mathbb{R}^D$가 밀도 함수 $p(\mathbf{z})$를 가지는 확률 변수이고, $f: \mathbb{R}^D \rightarrow \mathbb{R}^D$가 미분 가능한 단사함수일 때, $\mathbf{z}$의 밀도 함수 $p_{\mathbf{z}}$와 $\mathbf{x} = f(\mathbf{z})$의 밀도 함수 $p_{\mathbf{x}}$ 사이에는 다음과 같은 관계가 있다.
$$
p_{\mathbf{z}}(\mathbf{z}) = \left| \det \frac{\partial f}{\partial \mathbf{z}}(\mathbf{z}) \right| p_{\mathbf{x}}(\mathbf{x})
$$
여기서 $\frac{\partial f}{\partial \mathbf{z}}$는 $f$의 Jacobian 행렬이다. 따라서 $\mathbf{x}$의 밀도 함수는 다음과 같다.
$$
p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f}{\partial \mathbf{z}}(f^{-1}(\mathbf{x})) \right|^{-1}
$$
{{< /callout >}}

Change of variables formula는 같은 차원의 연속 확률 변수 사이 일대일 대응이 존재할 때 활용할 수 있다. 적분의 변수 변환과 정확히 같은 형태이다.

이제 위 식을 활용해 $\mathbf{x}$의 밀도 함수를 구하면 다음과 같다.
$$
p_{\theta}(\mathbf{x}) = p(f_{\theta}^{-1}(\mathbf{x})) \left| \det \frac{\partial f_{\theta}}{\partial \mathbf{z}}(f_{\theta}^{-1}(\mathbf{x})) \right|^{-1}
$$

{{< toggle title="Dirac delta function을 이용한 접근">}}
Dirac delta function을 이용해서도 같은 결과를 얻을 수 있다. 다음과 같은 [Dirac delta function의 성질](https://en.wikipedia.org/wiki/Dirac_delta_function#Properties_in_n_dimensions)을 이용하면 된다.

{{< callout type="Property">}}
$g: \mathbb{R}^D \rightarrow \mathbb{R}^D$이 미분 가능한 단사함수이면, 임의의 함수 $f$에 대해 다음이 성립한다.
$$
\int_{\mathbb{R}^D} f(\mathbf{z}) \, \delta(g(\mathbf{z})) \, d\mathbf{z} = \frac{f(g^{-1}(\mathbf{0}))}{\left| \det \frac{\partial g}{\partial \mathbf{z}}(g^{-1}(\mathbf{0})) \right|}
$$
여기서 $\frac{\partial g}{\partial \mathbf{z}}$는 $g$의 Jacobian 행렬이다.
{{< /callout >}}
이 성질을 식 {{< eqref wrong-trial-with-delta >}}에 적용하자. $g(\mathbf{z}) = \mathbf{x} - f_{\theta}(\mathbf{z})$로 놓으면, $g^{-1}(\mathbf{0}) = f_{\theta}^{-1}(\mathbf{x})$이고 $\frac{\partial g}{\partial \mathbf{z}} = -\frac{\partial f_{\theta}}{\partial \mathbf{z}}$이므로 $\left|\det \frac{\partial g}{\partial \mathbf{z}}\right| = \left|\det \frac{\partial f_{\theta}}{\partial \mathbf{z}}\right|$이다. 따라서,

$$
p_{\theta}(\mathbf{x}) = \int p(\mathbf{z}) \, \delta(\mathbf{x} - f_{\theta}(\mathbf{z})) \, d\mathbf{z} = \frac{p(f_{\theta}^{-1}(\mathbf{x}))}{\left|\det \frac{\partial f_{\theta}}{\partial \mathbf{z}}(f_{\theta}^{-1}(\mathbf{x}))\right|}
$$

이는 change of variables formula와 동일한 결과이다. 사실 위 성질 자체가 change of variables formula를 통해 유도한 것이다.
{{< /toggle >}}

이제 밀도 함수를 알았으니까 KL divergence 최소화 등의 최적화 문제를 풀 수 있을까? 여기에서 몇 가지 문제가 발생하는데, 먼저 $f_{\theta}$의 역함수를 구할 수 있어야 하는데, 이것부터 대부분의 신경망이 만족하지 못하는 조건이다. 그리고 $f_{\theta}$의 Jacobian determinant를 구하기는 더 어렵다. Jacobian은 $D \times D$ 행렬이고, 이를 backpropagation으로 직접 계산하기 위해서는 $f_{\theta}$의 각 출력 인자를 각 parameter로 미분해야 한다. 결국 $D \times (\text{parameter 수})$ 만큼의 미분값이 필요하고, 실제 연산 횟수는 더 필요하다. 차원 $D$가 크거나 parameter 수가 많으면 극도로 비효율적이다. 결국 이 방법을 활용하기 위해서는 $f_{\theta}$를 특수한 형태로 제한할 수밖에 없다. 다음 포스트에서 살펴볼 normalizing flow가 이러한 방법을 사용한다.

{{< callout type="Note" >}}
여기까지만 보면 이 접근이 까다로워 보이지만, 이렇게 해서도 강력한 모델을 만들 수 있다. Normalizing flow를 일반화한 neural ODE (NODE)나, 이 시리즈의 최종 목표인 flow matching도 결국 change of variables formula에서 출발한다.
{{< /callout >}}

### Case 2: $d < D$이거나, decoder의 역함수가 존재하지 않음

사실 latent variable model에서 $d < D$라는 선택은 매우 자연스럽다. 우리가 잠재 변수를 도입한 목적은 복잡한 관측 데이터에서 상대적으로 단순한 구조를 포착하는 것이었고, 이는 latent space의 차원이 data space보다 낮아야 의미가 있기 때문이다. 또한 $d = D$이더라도 일반적인 신경망 $f_{\theta}$가 단사함수라는 보장은 없다.

이 경우 change of variables formula를 적용할 수 없다. 특히 $d < D$이면, $f_{\theta}$의 치역 $S$는 $\mathbb{R}^D$ 안의 $d$차원 manifold이므로 $\mathbb{R}^D$에서의 Lebesgue 측도가 $0$이다. 즉, $p_{\theta}(\mathbf{x})$를 $\mathbb{R}^D$ 위의 **밀도 함수로 나타낼 수 없다**. 따라서 $p_{\theta}(\mathbf{x})$를 명시적으로 계산해야 하는 KL divergence 최소화나 maximum likelihood 같은 방법은 사용할 수 없다.

결국 $p_{\theta}(\mathbf{x})$를 명시적으로 나타내지 않고 최적화하거나 (GAN), 아예 밀도 함수를 무시하고 학습해야 한다 (autoencoder). 이것이 이번 포스트에서 살펴볼 방법들이다.

{{< callout type="Note" >}}
VAE에서는 $d < D$가 되어도 상관없었다. 왜냐하면 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 정규 분포 등 연속적인 분포로 정의하여 $d$차원 manifold 밖으로도 확률이 퍼지기 때문이다. 이때 $p_{\theta}(\mathbf{x})$는 $\mathbb{R}^D$ 위에서 올바르게 정의된 밀도 함수가 된다.
{{< /callout >}}

{{< callout type="Note" >}}
사실 $p_{\theta}(\mathbf{x})$뿐만 아니라 $p_{\mathrm{data}}(\mathbf{x})$도 같은 문제를 가지고 있다. 사진을 예로 들면, 각 픽셀이 독립적으로 값을 가지는 것이 아니라 물체의 형태, 조명, 질감 등 소수의 요인에 의해 결정된다. 따라서 가능한 모든 픽셀 조합의 공간 $\mathbb{R}^D$에 비해 실제 자연 이미지가 차지하는 영역은 극히 일부이다. 이처럼 실제 데이터가 $\mathbb{R}^D$ 안의 저차원 manifold 위에 집중되어 있다는 가설을 [manifold hypothesis](https://en.wikipedia.org/wiki/Manifold_hypothesis)라고 하며, 이는 기계 학습이 복잡한 고차원 데이터를 다룰 수 있는 본질적인 이유이다. Manifold hypothesis가 성립하면 $p_{\mathrm{data}}(\mathbf{x})$도 $\mathbb{R}^D$ 위의 밀도 함수로 나타낼 수 없다.

지금까지는 이 문제를 의식하지 않아도 상관없었다. 지금까지 $p_{\mathrm{data}}$에 접근하는 방식은 데이터셋에서 샘플을 뽑는 것뿐이었고, $p_{\mathrm{data}}(\mathbf{x})$의 값을 직접 계산할 필요가 없었기 때문이다. 하지만 이후 score-based model을 다룰 때, $p_{\mathrm{data}}$가 manifold 위에 집중되어 있다는 사실이 실제로 문제를 일으키게 된다.
{{< /callout >}}

$d > D$인 경우는 별로 유용하지 않으므로 넘어가자.

# Autoencoders와 PCA

Autoencoder는 입력 $\mathbf{x}$를 encoder로 저차원 표현 $\mathbf{z}$로 압축한 뒤, decoder로 $\mathbf{z}$에서 $\mathbf{x}$를 복원하는 모델이다. $d < D$임에도 불구하고 복원이 잘 되도록 학습하면, $\mathbf{z}$에는 자연스럽게 데이터의 핵심적인 구조가 담기게 된다.

Autoencoder라는 이름의 모델은 특정 논문에서 처음 제안된 것은 아니며, **PCA (Principal Component Analysis)** 등의 기법에서 자연스럽게 발전한 개념이다. 1986년 Rumelhart, Hinton, Williams의 유명한 backpropagation 논문{{< ref 2 >}}에서도 신경망을 이용해 입력을 압축하고 복원하는 구조가 등장한다. PCA는 다양한 분야에서 활용되고 이론적으로도 매우 아름다운 알고리즘이며, autoencoder는 단순히 PCA에 신경망을 적용한 것으로 볼 수 있다. 따라서 이 포스트에서는 PCA에 훨씬 초점을 맞추어 설명했다.

Autoencoder는 latent space에서 $\mathbf{z}$가 어떤 분포를 따르는지에 대해서는 전혀 신경쓰지 않는다. 따라서 학습된 decoder가 있더라도 latent space에서 $\mathbf{z}$를 샘플링할 방법이 마땅치 않다. 데이터의 분포가 encoder를 통해 효과적으로 압축되었다면 임의의 $\mathbf{z}$를 decoding해 의미 있는 $\mathbf{x}$를 샘플링할 가능성도 있지만, 일반적으로는 생성 모델로 사용하기 어렵다.

## SVD와 Low Rank Approximation

먼저 PCA의 핵심 도구인 **SVD (Singular Value Decomposition)** 에 대해 살펴보자.

{{< callout type="SVD (Singular Value Decomposition)">}}
모든 $m \times n$ 행렬 $\mathbf{A}$는 다음과 같이 세 행렬의 곱으로 분해할 수 있다.
$$
\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{\intercal}
$$
여기서 $\mathbf{U} \in \mathbb{R}^{m \times m}$과 $\mathbf{V} \in \mathbb{R}^{n \times n}$는 직교 행렬(orthogonal matrix)이다. $\boldsymbol{\Sigma} \in \mathbb{R}^{m \times n}$는 대각 성분을 제외한 모든 원소가 $0$고, $\min(n, m)$개 존재하는 대각 성분은 모두 음이 아닌 실수이다. $\Sigma$의 대각 성분을 **singular value**라고 하며, 크기 순서대로 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(n, m)} \geq 0$ 로 쓴다. $\mathbf{A}$의 singular value들은 유일하게 결정된다. 
{{< /callout >}}

지금의 맥락에서 SVD가 등장하는 이유는, SVD를 이용해 어떤 행렬을 더 낮은 rank의 행렬로 근사할 수 있기 때문이다. $m \times n$ 행렬 $\mathbf{A}$를 표현하기 위해서는 $mn$개의 값이 필요하지만, 이 행렬을 rank가 $d \ll mn$인 행렬로 근사할 수 있다면 훨씬 적은 개수의 값을 통해 $\mathbf{A}$를 근사적으로 표현할 수 있다. 이 방법을 구체적으로 살펴보자.

$\mathbf{V}$의 열벡터를 $\mathbf{v}_1, \ldots, \mathbf{v}_n$, $\mathbf{U}$의 열벡터를 $\mathbf{u}_1, \ldots, \mathbf{u}_m$이라 하자. 그럼 $\mathbf{A}$를 다음과 같이 쓸 수 있다.

$$
\mathbf{A} = \sum_{i=1}^{\min(m, n)} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\intercal}
$$

위 식에서 $\mathbf{u}_i \mathbf{v}_i^{\intercal}$는 두 벡터의 곱으로 만들어진 $m \times n$ 행렬이고, rank는 1이다. 즉, 위 표현은 $\mathbf{A}$를 rank가 $1$인 행렬 $\min(n, m)$개의 합으로 분해하여 나타낸 것이다.

이 분해에서 각 항의 계수인 singular value $\sigma_i$에 주목하자. $\sigma_i$가 큰 항은 행렬에서 중요한 성분이고, 작은 항은 덜 중요한 성분이라고 직관적으로 이해할 수 있다. Singular value들은 이미 크기 순서대로 정렬되어 있으므로, $i$가 커질수록 $\sigma_{i} \mathbf{u}_i \mathbf{v}_i^{\intercal}$가 $\mathbf{A}$에서 덜 중요하다. 그렇다면, 상위 $d$개의 항만 남기면 행렬의 핵심적인 구조를 $d$개의 성분으로 압축할 수 있지 않을까?

$$
\mathbf{A}_d = \sum_{i=1}^{d} \sigma_i \mathbf{u}_i \mathbf{v}_i^{\intercal} = \mathbf{U}_d \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}
$$

여기서 $\mathbf{U}_d \in \mathbb{R}^{m \times d}$, $\mathbf{V}_d \in \mathbb{R}^{n \times d}$는 각각 $\mathbf{U}$, $\mathbf{V}$에서 처음 $d$개의 열만 취한 것이고, $\boldsymbol{\Sigma}_d \in \mathbb{R}^{d \times d}$는 $\boldsymbol{\Sigma}$에서 $d$개의 행과 열을 취해 가장 큰 $d$개의 singular value만 남긴 것이다. 이 세 행렬의 곱으로 정의되는 행렬 $\mathbf{A}_{d}$는 rank가 최대 $d$인 행렬이다. $\mathbf{U}_{d}$, $\mathbf{V}_{d}$, $\boldsymbol{\Sigma}_{d}$에는 각각 최대 $md$, $nd$, $d$ 개의 $0$이 아닌 값이 있으므로, $\mathbf{A}_{d}$는 $(m + n + 1)d$ 개의 값으로 표현할 수 있는 행렬이다. $d \ll mn$인 경우 이는 원래의 행렬 $\mathbf{A}$가 가지고 있던 $mn$ 개의 값보다 훨씬 적은 수의 값이다. (사실 $\mathbf{U}_{d}$와 $\boldsymbol{\Sigma}_{d}$를 합치면 $(m + n)d$ 개의 값으로도 충분하다. $\mathbf{U}_{d}\boldsymbol{\Sigma}_{d}$는 $m \times d$ 행렬이기 때문이다.)

지금까지는 직관에 기대어 $\mathbf{A}_{d}$를 정의했는데, 이것이 유용하려면 이 행렬이 실제로 $\mathbf{A}$를 잘 근사해야 할 것이다. 이는 다음 정리를 통해 설명할 수 있다. 따라서 $\mathbf{A}_d$를 $\mathbf{A}$의 **rank-$d$ approximation**이라 한다.

{{< callout type="Eckart–Young–Mirsky Theorem" >}}
$\mathbf{A}$가 $m \times n$ 행렬일 때, $\mathbf{A}_d$는 rank가 $d$ 이하인 $m \times n$ 행렬 중에서 $\mathbf{A}$와의 Frobenius norm 차이를 최소화하는 행렬이다. 즉,
$$
\mathbf{A}_d = \argmin_{\text{rank}(\mathbf{B}) \leq d} \lVert \mathbf{A} - \mathbf{B} \rVert_F
$$
{{< /callout >}}

{{< callout type="Note" >}}
행렬 $\mathbf{A} \in \mathbb{R}^{m \times n}$의 **Frobenius norm**은 모든 원소의 제곱합의 제곱근으로 정의된다.
$$
\lVert \mathbf{A} \rVert_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}
$$
{{< /callout >}}

{{< toggle title="Eckart–Young–Mirsky theorem의 증명 (by Claude)" >}}
$r = \min(m, n)$으로 놓자. 증명은 세 단계로 이루어진다.

**Step 0. 준비: Frobenius norm의 직교 불변성.** Frobenius norm에 대해 $\|\mathbf{M}\|_F^2 = \mathrm{tr}(\mathbf{M}^{\intercal}\mathbf{M})$가 성립하므로, 직교 행렬 $\mathbf{P}$에 대해

$$\|\mathbf{PM}\|_F^2 = \mathrm{tr}(\mathbf{M}^{\intercal}\mathbf{P}^{\intercal}\mathbf{P}\mathbf{M}) = \mathrm{tr}(\mathbf{M}^{\intercal}\mathbf{M}) = \|\mathbf{M}\|_F^2$$

이 성립한다. 같은 방법으로 $\|\mathbf{MP}\|_F = \|\mathbf{M}\|_F$도 성립한다. 즉, **Frobenius norm은 직교 행렬을 곱해도 변하지 않는다.** 이 성질은 이후 단계에서 핵심적으로 사용된다.

**Step 1. $\mathbf{A}_d$의 오차 계산.** 먼저 $\|\mathbf{A} - \mathbf{A}_d\|_F^2$의 값을 구한다. $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\intercal}$에서 양쪽에 $\mathbf{U}^{\intercal}$과 $\mathbf{V}$를 곱하면 대각 행렬 $\boldsymbol{\Sigma}$가 되므로, 문제가 대각 행렬의 문제로 단순화된다. 구체적으로, Step 0의 직교 불변 성질에 의해

$$\|\mathbf{A} - \mathbf{A}_d\|_F^2 = \|\mathbf{U}^{\intercal}(\mathbf{A} - \mathbf{A}_d)\mathbf{V}\|_F^2 = \|\boldsymbol{\Sigma} - \mathbf{U}^{\intercal}\mathbf{A}_d\mathbf{V}\|_F^2$$

이다. $\mathbf{U}_d$는 $\mathbf{U}$의 처음 $d$개 열이므로 $\mathbf{U}^{\intercal}\mathbf{U}_d = \begin{bmatrix} \mathbf{I}_d \\\ \mathbf{0} \end{bmatrix}$이고, 마찬가지로 $\mathbf{V}_d^{\intercal}\mathbf{V} = \begin{bmatrix} \mathbf{I}_d & \mathbf{0} \end{bmatrix}$이다. 따라서 $\mathbf{U}^{\intercal}\mathbf{A}_d\mathbf{V}$는 $\boldsymbol{\Sigma}$에서 상위 $d$개의 singular value만 남기고 나머지를 $0$으로 바꾼 행렬이다. 이를 $\boldsymbol{\Sigma}$에서 빼면 대각 성분이 $0, \ldots, 0, \sigma_{d+1}, \ldots, \sigma_r$인 행렬이 남으므로,

$$\|\mathbf{A} - \mathbf{A}_d\|_F^2 = \sum_{i=d+1}^{r} \sigma_i^2$$

즉, $\mathbf{A}_d$의 근사 오차는 버린 singular value들의 제곱합이다.

**Step 2. 하한 증명: kernel을 이용한 오차 분리.** 이제 핵심 단계이다. rank가 $d$ 이하인 **임의의** $m \times n$ 행렬 $\mathbf{B}$에 대해, $\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \sum_{i=d+1}^{r} \sigma_i^2$임을 보이면 된다.

핵심 아이디어는 다음과 같다. Rank가 낮은 행렬 $\mathbf{B}$는 반드시 큰 kernel을 가진다. Kernel에 속하는 방향에서는 $\mathbf{B}$가 아무 기여도 하지 못하므로, 이 방향들에서 $\mathbf{A} - \mathbf{B}$는 그냥 $\mathbf{A}$가 된다. 따라서 오차가 $\mathbf{A}$의 크기만큼 발생하게 된다.

$\text{rank}(\mathbf{B}) \leq d$이면 rank-nullity theorem에 의해 $\ker(\mathbf{B})$의 차원은 $\geq n - d$이다. $\ker(\mathbf{B})$에서 $n - d$개의 orthonormal 벡터 $\mathbf{q}_1, \ldots, \mathbf{q}_{n-d}$를 뽑고, 이를 $\mathbb{R}^n$의 orthonormal basis $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$으로 확장하자. $\mathbf{Q} = [\mathbf{q}_1, \ldots, \mathbf{q}_n]$은 직교 행렬이다. Step 0의 직교 불변 성질에 의해 $\|\mathbf{A} - \mathbf{B}\|_F^2 = \|(\mathbf{A} - \mathbf{B})\mathbf{Q}\|_F^2$이고, 이를 각 열의 크기의 합으로 분해하면,

$$
\|\mathbf{A} - \mathbf{B}\|_F^2 = \sum_{j=1}^{n} \|(\mathbf{A} - \mathbf{B})\mathbf{q}_j\|^2
$$

이 합에서 처음 $n - d$개의 항만 남겨도 값은 줄어들지 않는다. 그리고 $j \leq n - d$이면 $\mathbf{q}_j \in \ker(\mathbf{B})$이므로 $\mathbf{B}\mathbf{q}_j = \mathbf{0}$이다. 즉 이 방향들에서는 $\mathbf{B}$의 기여가 없어서 $(\mathbf{A} - \mathbf{B})\mathbf{q}_j = \mathbf{A}\mathbf{q}_j$가 된다. 따라서,

$$
\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \sum_{j=1}^{n-d} \|\mathbf{A}\mathbf{q}_j\|^2
$$

**Step 3. 하한 증명: singular value로의 변환.** Step 2에서 오차의 하한을 $n - d$개 방향에서의 $\mathbf{A}$의 크기로 바꾸었다. 이제 남은 질문은, **이 $n - d$개 방향에서 $\mathbf{A}$의 크기가 얼마나 작아질 수 있는가**이다.

$\mathbf{Q}_{n-d} = [\mathbf{q}_1, \ldots, \mathbf{q}_{n-d}]$로 놓으면, 위 식의 우변은 $\|\mathbf{A}\mathbf{Q}_{n-d}\|_F^2$이다. $\mathbf{A}$의 SVD를 이용해 이를 singular value들로 표현하자. $\mathbf{A}^{\intercal}\mathbf{A} = \mathbf{V}\boldsymbol{\Sigma}^{\intercal}\boldsymbol{\Sigma}\mathbf{V}^{\intercal}$이므로,

$$
\|\mathbf{A}\mathbf{Q}_{n-d}\|_F^2 = \mathrm{tr}(\mathbf{Q}_{n-d}^{\intercal}\mathbf{A}^{\intercal}\mathbf{A}\mathbf{Q}_{n-d}) = \mathrm{tr}(\mathbf{Q}_{n-d}^{\intercal}\mathbf{V}\boldsymbol{\Sigma}^{\intercal}\boldsymbol{\Sigma}\mathbf{V}^{\intercal}\mathbf{Q}_{n-d})
$$

$\mathbf{P} = \mathbf{V}^{\intercal}\mathbf{Q}_{n-d} \in \mathbb{R}^{n \times (n-d)}$로 치환하면, $\mathbf{V}$가 직교 행렬이므로 $\mathbf{P}^{\intercal}\mathbf{P} = \mathbf{I}_{n-d}$이다. $\mathbf{p}_i^{\intercal}$를 $\mathbf{P}$의 $i$번째 행이라 하면,

$$
\|\mathbf{A}\mathbf{Q}_{n-d}\|_F^2 = \mathrm{tr}(\mathbf{P}^{\intercal}\boldsymbol{\Sigma}^{\intercal}\boldsymbol{\Sigma}\mathbf{P}) = \sum_{i=1}^{r} \sigma_i^2 \|\mathbf{p}_i\|^2
$$

이 식은 kernel 방향들에서의 $\mathbf{A}$의 크기를, 각 singular value $\sigma_i$에 대한 가중합으로 나타낸 것이다. $\mathbf{P}^{\intercal}\mathbf{P} = \mathbf{I}_{n-d}$에서 $\sum_{i=1}^{n} \|\mathbf{p}_i\|^2 = n - d$이고, 각 $\|\mathbf{p}_i\|^2 \in [0, 1]$이다. 이 가중합을 최소화하려면 $\sigma_i^2$가 큰 항 (즉 상위 $d$개의 singular value)의 가중치를 $0$으로 만들고, 나머지에 가중치 $1$을 부여하면 된다. $\sigma_1^2 \geq \cdots \geq \sigma_r^2 \geq 0$이므로,

$$
\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \sum_{i=1}^{r} \sigma_i^2 \|\mathbf{p}_i\|^2 \geq \sum_{i=d+1}^{r} \sigma_i^2 = \|\mathbf{A} - \mathbf{A}_d\|_F^2
$$

직관적으로 정리하면, rank가 $d$ 이하인 행렬은 최대 $d$개의 방향만 커버할 수 있다. 따라서 나머지 $n - d$개의 방향에서는 $\mathbf{A}$와의 차이를 줄일 수 없고, 이 방향들에서의 오차는 최소한 $\sigma_{d+1}^2 + \cdots + \sigma_r^2$이 된다. $\mathbf{A}_d$는 가장 큰 $d$개의 singular value 방향을 정확히 커버하므로, 오차가 정확히 이 하한과 일치한다.
{{< /toggle >}}

## PCA의 관점 1. Reconstruction Error 최소화

PCA의 목표는 고차원 데이터를 저차원으로 표현하는 것이다. 이를 위해 rank가 큰 행렬을 rank가 작은 행렬로 근사하는 SVD의 아이디어를 사용할 수 있다.

($p_{\mathrm{data}}(\mathbf{x})$로부터 얻은) $N$ 개의 $D$ 차원 데이터 $\mathbf{x}^{(1)}, \cdots, \mathbf{x}^{(N)} \in \mathbb{R}^{D}$이 주어졌다고 하자. 우리의 목표는 이들을 잘 표현하는 $d$ 차원 잠재 변수 $\mathbf{z}^{(1)}, \cdots, \mathbf{z}^{(N)} \in \mathbb{R}^{d}$를 찾는 것이다. 잠재 변수가 데이터를 잘 표현하기 위해서는 $\mathbf{z}^{(i)}$를 통해 $\mathbf{x}^{(i)}$를 (근사적으로) 복원할 수 있어야 한다. 이때, 복원에는 가장 단순한 변환인 **선형 변환**을 활용할 것이다. 선형 변환은 행렬을 사용해 표현할 수 있으므로, 우리의 목표는
$$
\mathbf{W} \mathbf{z}^{(1)} \approx \mathbf{x}^{(1)}, \qquad \cdots, \qquad \mathbf{W} \mathbf{z}^{(N)} \approx \mathbf{x}^{(N)}
$$
을 만족하는 잠재 표현 $\mathbf{z}^{(1)}, \cdots, \mathbf{z}^{(N)}$과 이들이 공유하는 행렬 $\mathbf{W} \in \mathbb{R}^{D \times d}$를 찾는 것이 된다. 그런데 $\mathbf{W} \mathbf{z}^{(i)}$가 $\mathbf{x}^{(i)}$를 잘 근사한다는 것을 어떻게 정의하면 좋을까? 자연스러운 방법 중 하나는 두 벡터의 차이의 크기, 즉 오차의 크기를 최소화하는 것이다. 이를 reconstruction error라고 한다.
$$
\| \mathbf{W} \mathbf{z}^{(i)} - \mathbf{x}^{(i)} \|^{2} = \sum_{j = 1}^{D} \left((\mathbf{W} \mathbf{z}^{(i)} - \mathbf{x}^{(i)})_{j}\right)^{2}
$$

Reconstruction error를 모든 데이터에 대해 더하면 다음과 같다.
{{< eqlabel mse >}}
$$
\sum_{i=1}^{N}\sum_{j = 1}^{D} \left((\mathbf{W} \mathbf{z}^{(i)} - \mathbf{x}^{(i)})_{j}\right)^{2}
$$

이제 rank-$d$ approximation과 관련짓기 위해, 식 {{< eqref mse >}}를 Frobenius norm의 형태로 만들자. 두 행렬 $\mathbf{X} \in \mathbb{R}^{D \times N}$과 $\mathbf{Z} \in \mathbb{R}^{d \times N}$를 각각 다음과 같이 정의하자.
$$
\mathbf{X} = \begin{bmatrix} \mathbf{x}^{(1)} & \cdots & \mathbf{x}^{(N)} \end{bmatrix}, \qquad \mathbf{Z} = \begin{bmatrix} \mathbf{z}^{(1)} & \cdots & \mathbf{z}^{(N)} \end{bmatrix}
$$

그럼 식 {{< eqref mse >}}는 다음과 같이 Frobenuis norm으로 나타낼 수 있다.
$$
\sum_{i=1}^{N}\sum_{j = 1}^{D} \left((\mathbf{W} \mathbf{z}^{(i)} - \mathbf{x}^{(i)})_{j}\right)^{2} = \| \mathbf{W} \mathbf{Z} - \mathbf{X} \|^{2}_{F}
$$

여기에서 $\mathbf{WZ}$는 $D \times d$인 행렬과 $d \times N$인 행렬의 곱이므로, rank가 최대 $d$이다. 따라서, 위 식을 최소화하는 것은 rank가 $d$ 이하인 행렬을 이용해 $\mathbf{X}$를 근사하는 것으로 생각할 수 있다. 이제 $\mathbf{X}$의 SVD인 $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{\intercal}$을 생각하자. $\mathbf{X}$의 rank-$d$ approximation은 $\mathbf{U}_d \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}$이니, 다음과 같은 식을 생각하자.
$$
\mathbf{W} \mathbf{Z} = \mathbf{U}_d \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}
$$

우리는 $\mathbf{W}$와 $\mathbf{Z}$를 모두 어느 정도 자유롭게 결정할 수 있다. $\mathbf{U}_d$, $\boldsymbol{\Sigma}_d$, $\mathbf{V}_d$의 크기를 고려하면 $\mathbf{W} = \mathbf{U}_d \boldsymbol{\Sigma}_d$, $\mathbf{Z} = \mathbf{V}_{d}^{\intercal}$ 또는 $\mathbf{W} = \mathbf{U}_d$, $\mathbf{Z} = \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}$ 등의 선택이 가능하다. 심지어 $\mathbf{W} = \mathbf{U}_d \boldsymbol{\Sigma}_d^{1/2}$, $\mathbf{Z} = \boldsymbol{\Sigma}_d^{1/2} \mathbf{V}_d^{\intercal}$도 가능하다. 이 중에서 어떤 것을 선택해야 할까? 결론부터 말하면, $\mathbf{W} = \mathbf{U}_d$, $\mathbf{Z} = \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}$를 선택해야 한다. 두 가지 관점에서 이 선택의 이유를 살펴보자.

첫째, encoding이 자연스럽다. $\mathbf{W} = \mathbf{U}_d$로 놓으면 $\mathbf{W}$의 열이 $\mathbb{R}^{D}$에서 orthonormal한 $d$개의 벡터가 된다. 이때, $\mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_{d}$ 이므로 다음이 성립한다.
$$
\mathbf{X} = \mathbf{W} \mathbf{Z} \implies \mathbf{W}^{\intercal} \mathbf{X} = \mathbf{W}^{\intercal} \mathbf{W} \mathbf{Z} = \mathbf{Z}
$$

이는 모든 $i$에 대해 $\mathbf{z}^{(i)} = \mathbf{W}^{\intercal} \mathbf{x}^{(i)}$ 가 성립한다는 의미이다. 원래 행렬 $\mathbf{W}$는 $\mathbf{z}$가 주어질 때 $\mathbf{x}$를 decoding하기 위해 사용했는데, $\mathbf{W}^{\intercal}$를 사용하면 $\mathbf{x}$를 $\mathbf{z}$로 encoding할 수 있는 것이다. 따라서 새로운 데이터 $\tilde{\mathbf{x}}$가 주어졌을 때 encoding이 $\tilde{\mathbf{z}} = \mathbf{W}^{\intercal} \tilde{\mathbf{x}}$로 주어진다. 물론 $\mathbf{W} = \mathbf{U}_d \boldsymbol{\Sigma}_d$와 같이 선택해도 encoding에 필요한 행렬을 비슷하게 유도할 수 있지만, $\mathbf{U}_d$를 선택했을 때와 같이 자연스럽게 주어지지는 않는다.

둘째, 잠재 변수 $\mathbf{Z}$에 불필요한 제약이 생기지 않는다. 만약 $\mathbf{Z} = \mathbf{V}_d^{\intercal}$로 놓으면, $\mathbf{V}_d$가 직교 행렬이므로 $\mathbf{Z}$의 행벡터가 orthonormal하다는 제약이 생긴다. 새로운 데이터 $\tilde{\mathbf{x}}$를 인코딩하여 $\mathbf{Z}$에 열 $\tilde{\mathbf{z}}$를 추가했을 때 이 성질이 유지되리라는 보장이 없다. $\boldsymbol{\Sigma}_d$를 $\mathbf{Z}$ 쪽으로 흡수하면 이러한 제약이 사라진다.

지금까지의 내용을 정리해 보자. SVD를 활용하면 식 {{< eqref mse >}}를 최소화하는 $\mathbf{W}$와 $\mathbf{Z}$를 구할 수 있으며, $\mathbf{W}$를 통해 decoding, $\mathbf{W}^{\intercal}$를 통해 encoding을 수행할 수 있다.

또한, $\mathbf{W} = \mathbf{U}_{d}$라는 선택의 두 가지 이유를 알아보았다. 그런데 이 선택이 편리하다는 것은 알 수 있었지만, 이 선택이 왜 필수적인지 설명하지는 못했다. 다음 절에서 분산 최대화라는 다른 관점에서 PCA를 살펴보면, $\mathbf{W} = \mathbf{U}_{d}$가 자연스럽다는 사실을 알 수 있다.

## PCA의 관점 2. 분산 최대화

앞 절에서는 reconstruction error를 최소화하는 관점에서 PCA를 유도했다. 이번 절에서는 '분산 최대화' 라는 다른 관점에서 PCA를 유도한다. 이 관점에서는 $\mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_d$가 편의를 위한 선택이 아니라 문제의 제약 조건으로 자연스럽게 등장하게 된다.

$D$ 차원 데이터를 $d$ 차원으로 압축하는 과정을 데이터를 $d$ 차원 부분공간에 투영하는 것으로 생각하자. 이 $d$ 차원 부분공간의 orthonormal basis를 $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{d}\}$라 하고, 이를 $\mathbb{R}^{D}$ 전체의 orthonormal basis로 확장해 $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{D}\}$라 하자. 이때 어떤 데이터 $\mathbf{x} \in \mathbb{R}^{D}$를 부분공간에 투영하는 것은, $\mathbf{x}$를 $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{D}\}$에 대한 좌표인 $(x_{1}, \cdots, x_{D})$로 쓴 뒤, 앞쪽 $d$ 개의 좌표인 $(x_{1}, \cdots, x_{d})$만 남기는 것으로 생각할 수 있다.

방금 전 논의에서는 $d$ 차원 부분공간을 먼저 잡고, 이를 통해 orthonormal basis $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{D}\}$를 잡았다. 이제 순서를 반대로 해서 orthonormal basis $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{D}\}$를 먼저 잡고, 이 중에서 $d$ 개의 좌표를 골라 $d$ 차원 부분공간을 만드는 상황을 생각해 보자. 이때, 우리는 어떤 좌표를 고르고 어떤 좌표를 제거할지 결정할 수 있다. 좌표를 고르는 기준을 마련하자.

$N$ 개의 데이터 $\mathbf{x}^{(1)}, \cdots, \mathbf{x}^{(N)}$을 모두 이 basis에 대한 좌표로 나타내, $i$번째 데이터의 좌표를 $(x^{(i)}_{1}, \cdots, x^{(i)}_{D})$라 하자. 이때, 각 좌표별로 데이터의 분포를 살펴보자. $j$번째 좌표에서 데이터의 분포는 $N$ 개의 실수 $\{x^{(1)}_{j}, \cdots, x^{(N)}_{j}\}$ 이다. 이 $D$ 개의 좌표들 중 어떤 좌표에 대해서는 데이터의 분포가 상대적으로 넓게 퍼져 있을 것이고, 어떤 좌표에 대해서는 분포가 좁을 것이다. 즉, 데이터의 **분산**이 좌표마다 다르다.

데이터의 좌표 중 $d$ 개만 남기면 정보가 손실된다. 이때, 분산이 큰 좌표는 그 좌표가 나타내는 방향에서 데이터가 넓게 퍼져 있음을 의미하므로, 그 좌표에는 많은 정보가 담겨 있다. 반대로 분산이 작은 좌표에서는 데이터가 상대적으로 한 곳에 모여 있으므로, 버려도 손실이 적다. 따라서 분산이 가장 큰 $d$ 개의 좌표를 고르면 정보 손실을 최소화할 수 있다.

{{< callout type="Note" >}}
그런데 위 논의에서는 각 좌표의 분산을 독립적으로 비교했다. 이것이 가능한 이유를 살펴보자. $D$ 차원 데이터의 변동을 나타내는 것은 스칼라 분산이 아니라 $D \times D$ 공분산 행렬 $\mathbf{C}$이다. 그런데 orthonormal basis를 사용하면 공분산 행렬의 대각합(trace)이 각 좌표의 분산의 합으로 분해된다.

$$
\mathrm{tr}(\mathbf{C}) = \sum_{j=1}^{D} \mathrm{Var}(x_j)
$$

$\mathrm{tr}(\mathbf{C})$는 orthonormal basis의 선택에 무관한 상수이다. 따라서, 어떤 orthonormal basis를 사용하든 각 좌표의 분산의 합은 항상 같은 값이 된다. 이는 곧 분산이 큰 $d$ 개의 좌표를 고르는 것이 나머지 $D - d$ 개의 좌표의 분산의 합을 최소화하는 것과 동치임을 의미한다. 각 좌표의 분산을 독립적으로 비교하는 것이 정당화되는 이유이다.
{{< /callout >}}

이제 우리의 목표를 다시 생각해 보자. 우리는 데이터의 정보를 최대한 보존하는 $d$ 차원 부분공간을 찾고 싶은데, orthonormal basis를 먼저 정한 뒤 이 중에 분산이 가장 큰 $d$ 개의 좌표를 골라 부분공간을 구성하면 된다. 특히, orthonormal basis를 자유롭게 정할 수 있다면 좌표를 분산이 큰 순서대로 정렬할 수 있으므로, 앞에서 $d$ 개의 좌표를 골라도 된다. 이제 우리의 목표는 다음과 같다.
> 데이터 $\mathbf{x}^{(1)}, \cdots, \mathbf{x}^{(N)}$이 주어졌을 때, $\mathbb{R}^{D}$의 orthonormal basis $\{\mathbf{v}_{1}, \cdots, \mathbf{v}_{D}\}$를 잘 정해서 첫 $d$ 개의 좌표의 분산의 합을 최대화하자.

 편의상 데이터의 평균이 $\mathbf{0}$이라고 가정하자. 만약 평균이 $\mathbf{0}$이 아닐 경우, 데이터 전체에서 평균을 빼면 된다.

먼저 $d = 1$인 경우를 살펴보자. 단위 벡터 $\mathbf{w} \in \mathbb{R}^D$ ($\|\mathbf{w}\| = 1$)를 하나 고정하면, 데이터 $\mathbf{x}^{(i)}$를 $\mathbf{w}$ 방향으로 투영한 값은 스칼라 $\mathbf{w}^{\intercal} \mathbf{x}^{(i)}$이다. 이 투영값의 분산은

$$
\frac{1}{N} \sum_{i=1}^{N} (\mathbf{w}^{\intercal} \mathbf{x}^{(i)})^2 = \frac{1}{N} \mathbf{w}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{w}
$$

이다. 분산을 최대화하는 방향을 찾는 문제는 다음과 같다.

$$
\max_{\mathbf{w}} \; \mathbf{w}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^{\intercal} \mathbf{w} = 1
$$

여기서 $\|\mathbf{w}\| = 1$이라는 제약이 필요한 이유는, 제약이 없으면 $\mathbf{w}$의 크기를 키우는 것만으로 분산을 임의로 크게 만들 수 있기 때문이다. 우리가 찾고 싶은 것은 크기가 아니라 **방향**이므로, 크기를 $1$로 고정하는 것이 자연스럽다.

이 문제의 해는 $\mathbf{X} \mathbf{X}^{\intercal}$의 최대 고유값에 대응하는 고유벡터이다. $\mathbf{X}$의 SVD $\mathbf{X} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^{\intercal}$를 이용하면 $\mathbf{X} \mathbf{X}^{\intercal} = \mathbf{U} \boldsymbol{\Sigma} \boldsymbol{\Sigma}^{\intercal} \mathbf{U}^{\intercal}$이므로, $\mathbf{X} \mathbf{X}^{\intercal}$의 고유벡터는 $\mathbf{u}_1, \ldots, \mathbf{u}_D$이고 대응하는 고유값은 $\sigma_1^2, \ldots, \sigma_D^2$이다. 따라서 분산을 최대화하는 방향은 $\mathbf{w} = \mathbf{u}_1$이다.

{{< toggle title="Lagrange multiplier를 이용한 풀이" >}}
Lagrange multiplier $\lambda$를 도입하면, 최적해에서 다음 조건이 성립해야 한다.

$$
\frac{\partial}{\partial \mathbf{w}} \left[ \mathbf{w}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{w} - \lambda (\mathbf{w}^{\intercal} \mathbf{w} - 1) \right] = \mathbf{0} \implies \mathbf{X} \mathbf{X}^{\intercal} \mathbf{w} = \lambda \mathbf{w}
$$

즉, 최적의 $\mathbf{w}$는 $\mathbf{X} \mathbf{X}^{\intercal}$의 고유벡터이다. 이때 목적 함수의 값은 $\mathbf{w}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{w} = \lambda \mathbf{w}^{\intercal} \mathbf{w} = \lambda$이므로, 분산을 최대화하려면 가장 큰 고유값에 대응하는 고유벡터를 선택해야 한다.
{{< /toggle >}}

이제 $d$개의 방향을 동시에 찾는 문제를 생각하자. $\mathbf{W} = [\mathbf{w}_1, \ldots, \mathbf{w}_d] \in \mathbb{R}^{D \times d}$에 대해, 제약 $\mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_d$는 $\mathbf{w}_1, \ldots, \mathbf{w}_d$가 orthonormal하다는 조건이다. 투영된 데이터의 총 분산은

$$
\frac{1}{N} \sum_{i=1}^{N} \| \mathbf{W}^{\intercal} \mathbf{x}^{(i)} \|^2 = \frac{1}{N} \| \mathbf{W}^{\intercal} \mathbf{X} \|_F^2 = \frac{1}{N} \mathrm{tr}(\mathbf{W}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{W})
$$

이므로, 최적화 문제는 다음과 같다.

$$
\max_{\mathbf{W}} \; \mathrm{tr}(\mathbf{W}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{W}) \quad \text{s.t.} \quad \mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_d
$$

$\mathbf{A} = \mathbf{U}^{\intercal} \mathbf{W}$로 치환하면 $\mathbf{A}^{\intercal} \mathbf{A} = \mathbf{W}^{\intercal} \mathbf{U} \mathbf{U}^{\intercal} \mathbf{W} = \mathbf{I}_d$이고,

$$
\mathrm{tr}(\mathbf{W}^{\intercal} \mathbf{X} \mathbf{X}^{\intercal} \mathbf{W}) = \mathrm{tr}(\mathbf{A}^{\intercal} \boldsymbol{\Sigma} \boldsymbol{\Sigma}^{\intercal} \mathbf{A}) = \sum_{j=1}^{D} \sigma_j^2 \| \mathbf{a}_j \|^2
$$

여기서 $\mathbf{a}_j^{\intercal}$는 $\mathbf{A}$의 $j$번째 행이다. $\mathbf{A}^{\intercal} \mathbf{A} = \mathbf{I}_d$에서 $\sum_{j=1}^{D} \|\mathbf{a}_j\|^2 = d$이고, 각 $\|\mathbf{a}_j\|^2 \in [0, 1]$이다. $\sigma_1^2 \geq \cdots \geq \sigma_D^2 \geq 0$이므로, 이 가중합은 상위 $d$개 항에 가중치 $1$을 몰아줄 때 최대가 된다. 이때 $\mathbf{A} = \begin{bmatrix} \mathbf{I}_d \\ \mathbf{0} \end{bmatrix}$이고,

$$
\mathbf{W} = \mathbf{U} \mathbf{A} = \mathbf{U}_d
$$

이다. 즉, 분산 최대화 문제의 해는 $\mathbf{W} = \mathbf{U}_d$이다.

마지막으로, 분산 최대화와 reconstruction error 최소화가 동치임을 보이자. $\mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_d$일 때 $\mathbf{W} \mathbf{W}^{\intercal}$는 $\mathbf{W}$의 열 공간으로의 직교 투영 행렬이다. 직교 투영의 성질에 의해

$$
\| \mathbf{x}^{(i)} - \mathbf{W} \mathbf{W}^{\intercal} \mathbf{x}^{(i)} \|^2 = \| \mathbf{x}^{(i)} \|^2 - \| \mathbf{W}^{\intercal} \mathbf{x}^{(i)} \|^2
$$

이므로, reconstruction error는

$$
\frac{1}{N} \sum_{i=1}^{N} \| \mathbf{x}^{(i)} - \mathbf{W} \mathbf{W}^{\intercal} \mathbf{x}^{(i)} \|^2 = \frac{1}{N} \| \mathbf{X} \|_F^2 - \frac{1}{N} \| \mathbf{W}^{\intercal} \mathbf{X} \|_F^2
$$

로 분해된다. 첫 항은 $\mathbf{W}$에 무관하므로, reconstruction error를 최소화하는 것은 투영된 데이터의 분산을 최대화하는 것과 같다.

앞 절에서는 Eckart–Young–Mirsky 정리를 통해 $\mathbf{W} \mathbf{Z} = \mathbf{U}_d \boldsymbol{\Sigma}_d \mathbf{V}_d^{\intercal}$를 얻은 후, $\mathbf{W}$와 $\mathbf{Z}$를 어떻게 분리할지 선택해야 했다. 분산 최대화 관점에서는 처음부터 $\mathbf{W}^{\intercal} \mathbf{W} = \mathbf{I}_d$라는 제약이 문제에 포함되어 있으므로, $\mathbf{W} = \mathbf{U}_d$가 유일한 해로 도출된다. 또한, 최소 reconstruction error는 버려진 singular value들의 합

$$
\frac{1}{N} \sum_{j=d+1}^{D} \sigma_j^2
$$

으로 주어진다.

## Autoencoders

PCA의 encoder와 decoder는 각각 선형 변환 $\mathbf{W}^{\intercal}$과 $\mathbf{W}$로 제한되어 있다. 이를 통해 데이터를 $d$ 차원 부분공간으로 투영하여 근사할 수 있었다. 하지만 데이터가 선형 부분공간이 아닌 비선형 manifold 위에 있다면, 선형 투영만으로는 데이터의 구조를 효과적으로 포착할 수 없다. 이를 해결하는 자연스러운 방법은 encoder와 decoder를 더 표현력이 높은 함수로 대체하는 것이다. 특히, 딥러닝에서는 신경망을 활용할 수 있다.

Encoder $g_{\phi}: \mathbb{R}^D \rightarrow \mathbb{R}^d$와 decoder $f_{\theta}: \mathbb{R}^d \rightarrow \mathbb{R}^D$를 각각 신경망으로 정의하자. PCA에서와 마찬가지로, 목표는 encoding 후 decoding했을 때 원래 데이터를 잘 복원하는 것이다. 목적 함수는 다음과 같다.

{{< eqlabel ae-objective >}}
$$
\min_{\theta, \phi} \mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})} \left[ \| \mathbf{x} - f_{\theta}(g_{\phi}(\mathbf{x})) \|^2 \right]
$$

이는 PCA의 reconstruction error인 식 {{< eqref mse >}}와 동일한 형태의 식을 기댓값으로 표현한 것이다. 다만 PCA에서는 encoder와 decoder가 모두 행렬 $\mathbf{W}$로 결정되는 선형 함수였기 때문에 SVD를 통해 해석적인 최적해를 구할 수 있었지만, 신경망의 경우에는 경사 하강법으로 $\theta$와 $\phi$를 동시에 최적화해야 한다.

실제 학습에서는 기댓값을 직접 계산할 수 없으므로, 다음과 같이 몬테 카를로 근사를 사용한다.

$$
\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})} \left[ \| \mathbf{x} - f_{\theta}(g_{\phi}(\mathbf{x})) \|^2 \right] \approx \frac{1}{N} \sum_{n=1}^{N} \| \mathbf{x}^{(n)} - f_{\theta}(g_{\phi}(\mathbf{x}^{(n)})) \|^2
$$

이 근사된 목적 함수를 $\theta$와 $\phi$에 대해 미분해 gradient를 구하고, 이를 이용해 경사 하강법으로 최적화 문제를 풀 수 있다. 참고로, encoder와 decoder를 선형 변환으로 제한하면, 즉 $g_{\phi}(\mathbf{x}) = \mathbf{W}_1^T \mathbf{x}$, $f_{\theta}(\mathbf{z}) = \mathbf{W}_2 \mathbf{z}$로 놓으면, 이 목적 함수를 경사 하강법으로 학습시켰을 때 PCA에서 얻은 해와 동일한 결과가 나온다.

비선형 신경망을 사용하면 PCA보다 풍부한 표현을 학습할 수 있지만, autoencoder 자체는 생성 모델로 사용하기 어렵다. Decoder $f_{\theta}$가 잘 학습되었더라도, latent space에서 임의의 $\mathbf{z}$를 샘플링한 뒤 decoding했을 때 의미 있는 $\mathbf{x}$가 나온다는 보장이 없기 때문이다.

VAE는 이 문제를 다음과 같이 해결했다. 먼저, $\mathbf{z}$의 prior인 $p_{\theta}(\mathbf{x})$를 정의했다. Prior가 주어진 상태에서 계산 가능한 목적 함수를 얻기 위해 variational Bayes의 아이디어를 차용해 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 근사하는 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 도입했다. 이를 이용해 reconstruction term과 regularization term을 가진 목적 함수를 유도할 수 있었다 ([참고](../04-variational-autoencoder-2/#reconstruction-term과-regularization-term)). Autoencoder의 목적 함수인 식 {{< eqref ae-objective >}}는 reconstruction term과 같은 역할을 하며, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$와 $p_{\theta}(\mathbf{x})$의 KL divergence로 정의되는 regularization term이 $\mathbf{z}$의 분포를 제한하는 역할을 한다. 이렇게 하면 $\mathbf{z}$는 우리가 원하는 분포인 $p_{\theta}(\mathbf{z})$를 따르도록 학습되며, 반대로 $p_{\theta}(\mathbf{z})$에서 $\mathbf{z}$를 샘플링했을 때 이를 decoding해 의미 있는 $\mathbf{x}$를 얻을 수 있다.

이런 의미에서, VAE는 autoencoder를 발전시킨 것이라고도 볼 수 있다. 이때, $p_{\theta}(\mathbf{x} \mid \mathbf{z})$는 $\mathbf{z}$에 대응하는 $\mathbf{x}$를 찾는 decoder, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$는 $\mathbf{x}$에 대응하는 $\mathbf{z}$를 찾는 encoder로 볼 수 있다. 하지만 VAE의 목적 함수를 유도하기 위해서는 이러한 관점보다는 variational Bayes의 'posterior 근사' 관점이 더 자연스럽기 때문에, 이전 포스트에서는 이러한 관점을 최대한 배제하고 설명했다. 하지만 autoencoder의 관점도 VAE의 중요한 motivation이었고, VAE를 이해하는 데 큰 도움을 준다.

# Generative Adversarial Networks

앞서 deterministic한 decoder를 사용할 때 $\mathbf{x}$의 밀도 함수를 나타내기 어렵다는 사실을 확인했다. 심지어 $d < D$인 경우에는 $\mathbb{R}^{D}$에서 $\mathbf{x}$의 밀도 함수가 아예 존재하지 않았다. Generative adversarial network (GAN)은 adversarial nets라는 기발한 아이디어를 통해 **명시적인 밀도 함수 없이** 학습할 수 있다. 이것이 어떻게 가능할까?

## Generator와 Discriminator

GAN의 잠재 변수 $\mathbf{z} \in \mathbb{R}^{d}$는 prior $p(\mathbf{z})$를 따른다. GAN의 decoder는 신경망 $G_{\theta}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{D}$로 나타내며, **generator**라고 부른다. 또한 generator를 학습시키기 위해 또 다른 신경망 $D_{\phi}: \mathbb{R}^{D} \rightarrow (0, 1)$를 사용하는데, 이를 **discriminator**라고 부른다.

우리가 지금까지 살펴보았던 VAE나 autoencoder에서도 마찬가지로 두 개의 신경망을 이용해 학습했다. 여기에서 decoder와 쌍을 이루었던 신경망은 실제 데이터 $\mathbf{x}$를 잠재 변수 $\mathbf{z}$로 대응시키는 encoder였다. 한편 GAN에서 주어지는 discriminator의 경우, 정의역은 실제 데이터의 공간인 $\mathbb{R}^{D}$가 맞지만 공역은 $(0, 1)$이다. 따라서 학습 과정 또한 매우 다르다.

Discriminator의 역할을 직관적으로 설명하면, $\mathbf{x} \in \mathbb{R}^{D}$가 주어졌을 때 이것이 $p_{\mathrm{data}}$에서 샘플링한 것인지, 아니면 generator가 생성한 것인지 구분하는 역할을 한다. Discriminator가 보기에 $p_{\mathrm{data}}$에서 $\mathbf{x}$를 샘플링했을 가능성이 높다면 $1$에 가까운 출력을, 아니라면 $0$에 가까운 출력을 내놓는다.

자연스럽게 discriminator의 목표는 실제 데이터와 generator가 생성한 데이터를 잘 구분하는 것이 된다. 한편, generator의 목표는 실제 데이터에 가까운 분포를 따르는 데이터를 생성하는 것인데, 이는 discriminator가 구분해 내기 어려운 데이터를 생성하는 것으로 생각할 수 있다. 즉, generator와 discriminator는 서로의 목표와 상충하는 목표를 가지고 있다. 이를 **adversarial nets framework**라고 한다.

## GAN의 목적 함수

이제 이러한 adversarial nets framework에서 목적 함수를 구체적으로 어떻게 쓸 수 있는지 살펴보자. 먼저 discriminator의 목적 함수를 살펴보자. 지금까지 해 왔던 것처럼 MLE의 관점을 사용하자.

먼저 새로운 확률 변수 $y \in \{0, 1\}$을 도입해, 실제 데이터에는 $y = 1$, generator가 생성한 데이터에는 $y = 0$을 부여하자. 이때, $\mathbf{x}$가 주어진 $y$의 확률 분포는 다음과 같다. 만약 $\mathbf{x} \sim p_{\mathrm{data}}({\mathbf{x}})$라면
$$
P(y = 0 \mid \mathbf{x}) = 0, \qquad P(y = 1 \mid \mathbf{x}) = 1
$$

만약 $\mathbf{z} \sim p(\mathbf{z})$이고, $\mathbf{x} = G_{\theta}(\mathbf{z})$ 이면
$$
P(y = 0 \mid \mathbf{x}) = 1, \qquad P(y = 1 \mid \mathbf{x}) = 0
$$

이다. 이제 discriminator를 이용해 이 확률 분포를 근사적으로 모델링해 보자. 만약 $\mathbf{x}$가 주어졌을 때 $D_{\phi}(\mathbf{x}) = p$라면, 이는 다음과 같은 분포를 의미하는 것으로 정의한다.
$$
P_{\phi}(y = 0 \mid \mathbf{x}) = 1 - p, \qquad P_{\phi}(y = 1 \mid \mathbf{x}) = p
$$

이제 MLE를 적용하기 위해 log-likelihood를 구해 보자. 만약 $\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})$라면, $y = 1$이므로 log-likelihood는 다음과 같다. $$\log P_{\phi}(y \mid \mathbf{x}) = \log p = \log D_{\phi}(\mathbf{x})$$

만약 $\mathbf{z} \sim p(\mathbf{z})$이고 $\mathbf{x} = G_{\theta}(\mathbf{z})$라면, $y = 0$이므로 log-likelihood는 다음과 같다.
$$\log P_{\phi}(y \mid \mathbf{x}) = \log (1 - p) = \log (1 - D_{\phi}(\mathbf{x})) = \log (1 - D_{\phi}(G_{\theta}(\mathbf{z})))$$

이제 $\mathbf{x}$의 분포를 고려해 log-likelihood의 기댓값을 구하면 다음과 같다. $\mathbf{x}$가 실제 관측 데이터인 경우와 generator가 생성한 데이터인 경우에 대한 기댓값을 더해서 나타냈다.

$$
\mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ \log D_{\phi}(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ \log (1 - D_{\phi}(G_{\theta}(\mathbf{z}))) \right]
$$

Discriminator는 이 식을 $\phi$에 대해 최대화한다. 한편, generator의 목표는 discriminator를 속이는 것이므로, 같은 식을 $\theta$에 대해 최소화해야 한다. 두 목표를 합치면 다음과 같은 **minimax** 문제가 된다.

$$
\min_{\theta} \max_{\phi} \; \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ \log D_{\phi}(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ \log (1 - D_{\phi}(G_{\theta}(\mathbf{z}))) \right]
$$

## GAN의 학습

우리가 지금까지 살펴본 최적화 문제들은 최소화 또는 최대화로 방향이 정해져 있는 문제였다. 하지만 GAN에서 주어진 minimax 문제는 $\theta$에 대해서는 최소화, $\phi$에 대해서는 최대화이다. 이를 해결하기 위해서는 $\theta$와 $\phi$를 번갈아가며 최적화해야 한다. 목적 함수를 $J(\theta, \phi)$로 쓰면,

$$
J(\theta, \phi) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ \log D_{\phi}(\mathbf{x}) \right] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} \left[ \log (1 - D_{\phi}(G_{\theta}(\mathbf{z}))) \right]
$$

이다. 학습은 다음 두 단계를 충분히 많이 반복하는 방식으로 이루어진다.

1. **Discriminator 업데이트**: $\theta$를 고정하고, $J(\theta, \phi)$를 $\phi$에 대해 경사 상승법으로 업데이트한다.
2. **Generator 업데이트**: $\phi$를 고정하고, $J(\theta, \phi)$를 $\theta$에 대해 경사 하강법으로 업데이트한다.

실제로는 $J(\theta, \phi)$에 들어 있는 기댓값을 직접 계산할 수 없으므로, 각 단계에서 몬테 카를로 근사를 사용해 $\theta$ 또는 $\phi$에 대한 gradient를 계산해야 한다. 원래 논문{{< ref 1 >}}에서는 매 단계마다 discriminator를 $k$번 업데이트한 뒤 generator를 $1$번 업데이트하는 방법을 제안했다.

{{< callout type="Note" >}}
Generator를 업데이트할 때, $J(\theta, \phi)$에서 $\theta$에 의존하는 항은 $\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log (1 - D_{\phi}(G_{\theta}(\mathbf{z})))]$뿐이다. 그런데 학습 초기에는 generator가 거의 랜덤한 출력을 내놓으므로 discriminator가 이를 쉽게 구분하고, $D_{\phi}(G_{\theta}(\mathbf{z})) \approx 0$이 된다. 이 영역에서 $\log(1 - t)$의 기울기는 $0$에 가까우므로 generator의 gradient가 매우 작아져 학습이 느려진다. 이를 해결하기 위해, generator의 목적 함수를 $\log(1 - D_{\phi}(G_{\theta}(\mathbf{z})))$의 최소화 대신 $\log D_{\phi}(G_{\theta}(\mathbf{z}))$의 최대화로 대체하는 방법을 사용할 수 있다. $D_{\phi}(G_{\theta}(\mathbf{z})) \approx 0$인 영역에서 $-\log t$의 기울기는 매우 크므로, 학습 초기에 generator에게 강한 gradient를 제공한다.
{{< /callout >}}

## GAN의 정당성

GAN의 목적 함수는 상당히 직관적이고, 이렇게 학습한 generator는 실제 데이터와 구분하기 힘든 데이터를 생성할 것이라고 쉽게 짐작할 수 있다. 그런데 '실제 데이터와 구분하기 힘든 데이터 생성'과 '실제 데이터의 분포와 유사한 분포를 따르는 데이터 생성'은 분명히 구분해야 할 문제이다. 예를 들어, 실제 사진과 동일한 이미지 딱 한 장만 생성할 수 있는 이미지 생성 모델은 전자의 목표는 달성하지만 후자의 목표와는 어긋나 있고, 새로운 데이터를 생성하지 못하므로 생성 모델로써의 쓸모도 없다. 그래서 우리는 첫 포스트에서부터 후자를 생성 모델의 목표로 삼았다. 그렇다면 GAN은 우리의 목표를 달성하지 못하는 것일까?

놀랍게도, $G_{\theta}$와 $D_{\phi}$의 표현력이 충분하다면 generator가 생성하는 데이터의 분포와 실제 데이터의 분포 $p_{\mathrm{data}}$ 사이의 divergence가 $0$이 될 때가 minimax 문제의 해라는 것을 증명할 수 있다. 여기에서 사용되는 divergence는 우리가 지금까지 사용한 KL divergence가 아닌 다른 divergence이다. 지금부터 이 증명을 살펴보자.

그런데 앞에서 $d < D$인 경우 밀도 함수가 정의되지 않는다고 하지 않았나? 여기에서는 이러한 점을 생각하지 말고, 항상 $d = D$이고 generator의 역함수가 존재해 밀도 함수가 깔끔하게 정의되는 상황을 가정하자. 이 경우만 증명한다 하더라도 GAN의 정당성을 설명하기에는 충분하다.

증명은 generator를 고정한 상태로 최적의 discriminator를 구한 뒤, 이 최적의 discriminator를 사용해 최적의 generator를 찾는 순서로 이루어진다. 지금부터 generator $G_{\theta}$가 생성하는 데이터의 분포를 $p_{\theta}(\mathbf{x})$로 나타내자.

### 최적의 discriminator

Generator $G_{\theta}$가 고정되어 있을 때, $J(\theta, \phi)$를 최대화하는 discriminator를 구하자. 먼저, 앞에서 정의한 분포 $p_{\theta}(\mathbf{x})$를 이용해 $J$의 두 번째 기댓값을 $\mathbf{z}$가 아니라 $\mathbf{x}$에 대해 쓴다.
$$
\begin{align*}
J(\theta, \phi) &= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\log D_{\phi}(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log (1 - D_{\phi}(G_{\theta}(\mathbf{z}))) ]\\
&= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\log D_{\phi}(\mathbf{x})] + \mathbb{E}_{\mathbf{x} \sim p_{\theta}(\mathbf{x})} [\log (1 - D_{\phi}(\mathbf{x})) ]
\end{align*}
$$

핵심 아이디어는 $J$를 다음과 같이 적분으로 나타내 두 기댓값을 합치는 것이다.

$$
\begin{align*}
J(\theta, \phi)
&= \int p_{\mathrm{data}}(\mathbf{x}) \log D_{\phi}(\mathbf{x}) \, d\mathbf{x} + \int p_{\theta}(\mathbf{x}) \log (1 - D_{\phi}(\mathbf{x})) \, d\mathbf{x}\\
&= \int \left(p_{\mathrm{data}}(\mathbf{x}) \log D_{\phi}(\mathbf{x}) + p_{\theta}(\mathbf{x}) \log (1 - D_{\phi}(\mathbf{x})) \right) d\mathbf{x}\\
\end{align*}
$$

$D_{\phi}(\mathbf{x})$의 표현력이 충분히 높다면, 각 $\mathbf{x}$에 대해 피적분함수를 독립적으로 최대화할 수 있다. 즉, 각 $\mathbf{x}$에 대해 다음 식
$$p_{\mathrm{data}}(\mathbf{x}) \log D_{\phi}(\mathbf{x}) + p_{\theta}(\mathbf{x}) \log (1 - D_{\phi}(\mathbf{x}))$$
을 최대화하는 값을 $D_{\phi}(\mathbf{x})$로 정의하는 것이다. 이때, 최적의 discriminator는 다음과 같이 정의된다.

$$
D_{\phi^{*}}(\mathbf{x}) = \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})}
$$

이는 직관적으로 이해할 수 있는 결과인데, $p_{\mathrm{data}}(\mathbf{x})$가 $p_{\theta}(\mathbf{x})$에 비해 클수록 $D_{\phi^{*}}(\mathbf{x})$가 $1$에 가까워지고, 반대의 경우 $0$에 가까워진다. 구체적인 증명은 아래와 같다.

{{< toggle title="증명">}}
함수 $f: (0, 1) \rightarrow \mathbb{R}$를 $f(t) = a \log t + b \log (1 - t)$로 정의하자 ($a, b > 0$). $f'(t) = a / t - b / (1 - t)$이므로 $f'(t) = 0$을 풀면 $t = a / (a + b)$이다. $f''(t) = -a / t^2 - b / (1 - t)^2 < 0$이므로 이 점은 최댓값이다. 피적분함수에서 $a = p_{\mathrm{data}}(\mathbf{x})$, $b = p_{\theta}(\mathbf{x})$, $t = D_{\phi}(\mathbf{x})$로 놓으면 최적의 discriminator가 $D_{\phi^{*}}(\mathbf{x}) = p_{\mathrm{data}}(\mathbf{x}) / (p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x}))$임을 알 수 있다.
{{< /toggle >}}

### 최적의 generator
이제 $D_{\phi^{*}}$를 $J(\theta, \phi)$에 대입하면, generator만의 함수 $J'(\theta)$를 얻는다.

$$
\begin{align*}
J'(\theta) &= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ \log \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})} \right] + \mathbb{E}_{\mathbf{x} \sim p_{\theta}(\mathbf{x})} \left[ \log \frac{p_{\theta}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})} \right]
\end{align*}
$$

이 식은 복잡해 보이지만, 다음과 같이 KL divergence로 정리할 수 있다.

$$
\begin{align*}
J'(\theta) &= -\log 4 + D_{\mathrm{KL}}\left( p_{\mathrm{data}} \,\middle\|\, \frac{p_{\mathrm{data}} + p_{\theta}}{2} \right) + D_{\mathrm{KL}}\left( p_{\theta} \,\middle\|\, \frac{p_{\mathrm{data}} + p_{\theta}}{2} \right)
\end{align*}
$$

{{< toggle title="유도 과정" >}}
$J'(\theta)$의 첫 번째 항을 정리하자.
$$
\begin{align*}
&\mathbb{E}_{p_{\mathrm{data}}} \left[ \log \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})} \right] \\
&= \mathbb{E}_{p_{\mathrm{data}}} \left[ \log p_{\mathrm{data}}(\mathbf{x}) - \log \left(p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})\right)\right] \\
&= \mathbb{E}_{p_{\mathrm{data}}} \left[ \log p_{\mathrm{data}}(\mathbf{x}) - \log \left(\frac{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})}{2} \right) - \log 2 \right] \\
&= -\log 2 + D_{\mathrm{KL}}\left( p_{\mathrm{data}} \,\middle\|\, \frac{p_{\mathrm{data}} + p_{\theta}}{2} \right)
\end{align*}
$$
두 번째 항도 같은 방법으로 정리하면,
$$
\mathbb{E}_{p_{\theta}} \left[ \log \frac{p_{\theta}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})} \right] = -\log 2 + D_{\mathrm{KL}}\left( p_{\theta} \,\middle\|\, \frac{p_{\mathrm{data}} + p_{\theta}}{2} \right)
$$
두 항을 더하면 $J'(\theta) = -\log 4 + D_{\mathrm{KL}}(p_{\mathrm{data}} \| m) + D_{\mathrm{KL}}(p_{\theta} \| m)$이다. 여기서 $m = (p_{\mathrm{data}} + p_{\theta}) / 2$이다. 참고로, $m$을 적분하면 $1$이 되므로 $m$도 올바른 확률 분포이다.

$$
\int m(\mathbf{x})\,d\mathbf{x} = \int \frac{p_{\mathrm{data}}(\mathbf{x}) + p_{\theta}(\mathbf{x})}{2}\,d\mathbf{x} = \frac{1}{2}\left(\int p_{\mathrm{data}}(\mathbf{x})\,d\mathbf{x} + \int p_{\theta}(\mathbf{x})\,d\mathbf{x}\right) = 1
$$
{{< /toggle >}}

식을 이러한 형태로 정리한 이유는, 우변의 두 KL divergence의 합이 **Jensen–Shannon divergence (JSD)** 의 $2$배이기 때문이다. 따라서,

$$
J'(\theta) = -\log 4 + 2 \, \mathrm{JSD}(p_{\mathrm{data}} \| p_{\theta})
$$

{{< callout type="Note" >}}
두 분포 $p$, $q$의 **Jensen–Shannon divergence**는 다음과 같이 정의된다.
$$
\mathrm{JSD}(p \| q) = \frac{1}{2} D_{\mathrm{KL}}\left( p \,\middle\|\, \frac{p + q}{2} \right) + \frac{1}{2} D_{\mathrm{KL}}\left( q \,\middle\|\, \frac{p + q}{2} \right)
$$
KL divergence와 달리 JSD는 대칭이고 항상 유한한 값을 가진다.
{{< /callout >}}

{{< toggle title="JSD와 $f$-divergence" >}}
JSD는 $f$-divergence의 일종이다. $f(t) = t \log t - (t + 1) \log \frac{t + 1}{2}$로 놓으면 이를 확인할 수 있다. $u = p(\mathbf{x}) / q(\mathbf{x})$로 쓰면,
$$
\begin{align*}
D_f(p \| q) &= \mathbb{E}_{q} \left[ f\!\left(\frac{p(\mathbf{x})}{q(\mathbf{x})}\right) \right] \\
&= \mathbb{E}_{q} \left[ \frac{p(\mathbf{x})}{q(\mathbf{x})} \log \frac{p(\mathbf{x})}{q(\mathbf{x})} - \left(\frac{p(\mathbf{x})}{q(\mathbf{x})} + 1\right) \log \frac{p(\mathbf{x}) / q(\mathbf{x}) + 1}{2} \right] \\
&= \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} - \int (p(\mathbf{x}) + q(\mathbf{x})) \log \frac{p(\mathbf{x}) + q(\mathbf{x})}{2 q(\mathbf{x})} \, d\mathbf{x} \\
&= D_{\mathrm{KL}}(p \| q) - \int p(\mathbf{x}) \log \frac{p(\mathbf{x}) + q(\mathbf{x})}{2 q(\mathbf{x})} \, d\mathbf{x} - \int q(\mathbf{x}) \log \frac{p(\mathbf{x}) + q(\mathbf{x})}{2 q(\mathbf{x})} \, d\mathbf{x}
\end{align*}
$$
첫 항과 두 번째 항을 합치면,
$$
\int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} - \int p(\mathbf{x}) \log \frac{p(\mathbf{x}) + q(\mathbf{x})}{2 q(\mathbf{x})} \, d\mathbf{x} = \int p(\mathbf{x}) \log \frac{2 p(\mathbf{x})}{p(\mathbf{x}) + q(\mathbf{x})} \, d\mathbf{x} = D_{\mathrm{KL}}\left(p \,\middle\|\, \frac{p + q}{2}\right)
$$
세 번째 항은,
$$
-\int q(\mathbf{x}) \log \frac{p(\mathbf{x}) + q(\mathbf{x})}{2 q(\mathbf{x})} \, d\mathbf{x} = \int q(\mathbf{x}) \log \frac{2 q(\mathbf{x})}{p(\mathbf{x}) + q(\mathbf{x})} \, d\mathbf{x} = D_{\mathrm{KL}}\left(q \,\middle\|\, \frac{p + q}{2}\right)
$$
따라서 $D_f(p \| q) = D_{\mathrm{KL}}(p \| m) + D_{\mathrm{KL}}(q \| m) = 2 \, \mathrm{JSD}(p \| q)$이다. 여기서 $m = (p + q) / 2$이다.
{{< /toggle >}}

다른 divergence들과 마찬가지로, JSD는 항상 $0$ 이상이고, 두 분포가 (거의 모든 점에서) 같을 때에만 $0$이다. 따라서 $J'(\theta)$의 최솟값은 $-\log 4$이며, $p_{\theta} = p_{\mathrm{data}}$일 때 달성된다. 이때 최적의 discriminator는 모든 $\mathbf{x}$에 대해 $D_{\phi^{*}}(\mathbf{x}) = 1/2$이다. 즉, generator가 완벽하게 학습되면 discriminator는 실제 데이터와 생성된 데이터를 전혀 구분하지 못하게 된다.

## GAN의 한계

위의 이론적 결과는 이상적인 상황을 가정한 것이다. 하지만 실제로는 $G_{\theta}$와 $D_{\phi}$의 표현력이 제한되어 있고, 기댓값을 몬테 카를로로 근사해야 하며, 최적화 문제를 풀 때 $\theta$에 대한 경사 하강법과 $\phi$에 대한 경사 상승법을 번갈아 적용해야 하기 때문에 몇 가지 한계가 있다.

### 한계 1. 학습의 불안정성

Minimax 최적화는 일반적인 최소화 문제보다 훨씬 어렵다. Generator와 discriminator를 번갈아 업데이트하는 방식은 수렴을 보장하지 않으며, 최적화 과정에서 $\theta$나 $\phi$가 진동하거나 발산할 수 있다. 따라서 두 신경망 사이 균형을 맞추는 것이 중요하다.

만약 discriminator가 너무 강하면, generator의 매개변수 $\theta$를 조금 바꾸더라도 discriminator는 여전히 실제 데이터와 생성된 데이터를 쉽게 구분할 수 있다. 따라서 discriminator의 출력이 $0$에 매우 가까운 상태에서 거의 변하지 않으며, generator의 gradient는 $0$에 가까운 값을 가지고 제대로 된 학습이 진행되지 않는다. 이러한 문제는 주로 generator가 랜덤한 노이즈에 가까운 값을 생성하는 학습 초기에 발생한다.

반대로 discriminator가 너무 약해도 generator가 의미 있는 학습 신호를 받지 못한다. 이 균형을 맞추기 위해 학습률, 네트워크 구조 등의 hyperparameter를 세심하게 조정해야 한다.

### 한계 2. Mode collapse

Generator가 $p_{\mathrm{data}}$의 전체 분포를 학습하지 못하고, 일부 mode에 집중해 생성하는 현상이다. 예를 들어, 숫자 이미지를 생성하는 GAN이 $0$부터 $9$까지 다양한 숫자를 생성하는 대신, discriminator를 가장 잘 속일 수 있는 특정 숫자만 반복적으로 생성할 수 있다.

이는 이론과 현실의 괴리 때문에 발생한다. 앞에서 GAN의 이론적인 정당성을 보일 때는 최적의 discriminator를 구한 뒤, 이를 이용해 generator를 최적화했다. 하지만 실제로는 discriminator가 매 단계에서 최적이 아닌 상태에서 generator의 최적화를 수행하게 된다. 이때 generator는 현재 discriminator의 약점을 파고드는 방향으로 학습되므로, 특정 mode에 집중하는 현상이 일어난다.

이러한 두 가지 한계점은 근본적으로 GAN의 학습 과정이 불안정하고 까다롭다는 것을 의미한다. 이들을 극복할 수 있는 방법이 많이 연구되었음에도 불구하고, 한때 생성 모델의 표준처럼 사용되던 GAN은 현재 diffusion 기반의 모델들에게 자리를 내주게 되었다. 하지만 GAN은 다른 생성 모델과 결합하거나 생성 모델의 성능 개선을 위한 보조 장치로써 여전히 유용하다 {{< ref 3 >}}.

# 정리

이 포스트에서는 먼저 deterministic decoder를 사용하는 생성 모델이 겪는 어려움을 살펴보았다. 먼저, $d = D$이고 decoder의 역함수가 존재하는 경우에는 change of variables formula를 통해 $\mathbf{x}$의 밀도 함수를 얻을 수 있지만, decoder의 역함수가 존재하고 Jacobian determinant를 계산할 수 있어야 한다는 어려움이 생긴다. 다음 포스트에서는 이를 극복한 normalizing flow에 대해 살펴볼 것이다.

$d < D$이거나 decoder의 역함수가 존재하지 않는 경우, $\mathbf{x}$의 밀도 함수 자체가 정의되지 않으므로 이를 우회해야 한다. 이 포스트에서는 두 가지 생성 모델을 살펴보았다. 첫 번째로 살펴본 autoencoder는 encoder와 decoder를 각각 신경망으로 모델링한 뒤, reconstruction error를 최소화하는 목적 함수를 이용해 학습한다. 이렇게 하면 잠재 변수 $\mathbf{z}$의 확률 분포를 무시하기 때문에 제대로 된 생성 모델을 얻을 수 없었다. 두 번째로 살펴본 GAN은 generator와 discriminator가 서로 상충하는 목표를 가진 adversarial nets framework를 통해 생성 모델을 학습한다. 이상적인 경우 generator는 실제 데이터와 동일한 분포의 데이터를 생성하도록 학습하지만, 현실과의 괴리에 따라 몇 가지 문제점이 발생한다.

이 포스트에서는 문제점 위주로 살펴보았음에도 불구하고, 이 모델들의 아이디어는 매우 유용하다. 특히, VAE와 비교했을 때는 decoder가 확률 분포가 아니기 때문에 선명한 결과물을 제공한다는 장점이 있다.

{{< reflist >}}
{{< refitem 1 >}}
Goodfellow, Ian, Pouget-Abadie, Jean, Mirza, Mehdi, Xu, Bing, Warde-Farley, David, Ozair, Sherjil, Courville, Aaron, and Bengio, Yoshua. "[Generative adversarial nets](https://arxiv.org/abs/1406.2661)". *NeurIPS*, 2014.
{{< /refitem >}}
{{< refitem 2 >}}
Rumelhart, David E., Hinton, Geoffrey E., and Williams, Ronald J. "[Learning representations by back-propagating errors](https://doi.org/10.1038/323533a0)". *Nature*, 323(6088): 533–536, 1986.
{{< /refitem >}}
{{< refitem 3 >}}
Lai, Chieh-Hsin, Song, Yang, Kim, Dongjun, Mitsufuji, Yuki, and Ermon, Stefano. "[The principles of diffusion models](https://arxiv.org/abs/2510.21890)". *arXiv preprint*, 2025.
{{< /refitem >}}
* 이 시리즈의 전반적인 내용을 참고했다.
{{< /reflist >}}