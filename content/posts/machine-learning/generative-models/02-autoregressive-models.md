---
title: "Generative Models - 02. Autoregressive Models"
date: 2026-02-11
tags: ["machine-learning", "generative-models"]
draft: false
---

*이 글은 Claude Opus 4.6의 도움을 받아 작성했다.*

첫 번째로 살펴볼 생성 모델은 autoregressive model이다. [지난 포스트](../01-overview/#idea-1-autoregressive-generation)에서 이미 autoregressive model에 대해서 간단히 살펴보았는데, 이 내용을 다시 정리하고 모델을 어떻게 학습시키는지까지 알아보자.

# Autoregressive Model의 아이디어

Autoregressive model은 다음과 같은 핵심 아이디어에 기반하고 있다.

{{< callout type="Idea" >}}
복잡한 확률 분포를 단순한 확률 분포들의 조합으로 분해하자. 분해된 각 분포가 충분히 단순하다면, 정규화와 샘플링 문제를 쉽게 해결할 수 있다.
{{< /callout >}}

데이터를 $\mathbf{x} = (x_{1}, \cdots, x_{L})$의 형태로 쓸 수 있다고 하자. 우리의 목표는 $\mathbf{x}$의 분포를 $x_{1}$의 분포, $x_{2}$의 분포, ... $x_{L}$의 분포로 분리하는 것이다. 이 $L$개의 원소들이 서로 독립적이라는 보장은 없기 때문에, 이 과정에서 자연스럽게 조건부 확률이 등장하게 된다.

실제 데이터 $\mathbf{x}$의 확률 분포 $p_{\mathrm{data}}({\mathbf{x}})$는 연쇄 법칙에 의해 다음과 같이 $L$개의 조건부 확률 분포들의 곱으로 표현할 수 있다. $x_{1}, \cdots, x_{i - 1}$을 $x_{1:i-1}$로 간단하게 표현했다. 또한, 편의상 $p_{\theta}(x_{1})$을 $p_{\theta}(x_{1} \mid x_{1:0})$으로 표기했다.

$$
\begin{align*}
p_{\mathrm{data}}(\mathbf{x}) &= p_{\mathrm{data}}(x_{1}) \prod_{i=2}^{L} p_{\mathrm{data}}(x_{i} \mid x_{1}, \cdots, x_{i - 1})\\
&= \prod_{i=1}^{L} p_{\mathrm{data}}(x_{i} \mid x_{1:i-1})
\end{align*}
$$

이를 이용하면, 매개화된 확률 분포 $p_{\theta}(\mathbf{x})$를 다음과 같이 구성할 수 있다.

{{< eqlabel p_theta-conditional >}}
$$p_{\theta}(\mathbf{x}) = \prod_{i=1}^{L} p_{\theta}(x_{i} \mid x_{1:i-1})$$

이전 포스트에서, 정규화 조건을 만족하고 샘플링이 가능한 분포를 만들기가 어렵다는 논의를 했다. Autoregressive model은 이 문제를 쉽게 해결한다. 개별적인 원소들의 분포 $p_{\theta}(x_{i} \mid x_{1:i-1})$가 정규화 조건을 만족하고 샘플링이 가능하면 전체 분포 $p_{\theta}(\mathbf{x})$도 마찬가지이기 때문이다. 그럼 이제 $p_{\theta}(x_{i} \mid x_{1:i-1})$를 다루기 쉬운 단순한 분포로 설정하면 된다.

다음과 같은 $L$ 번의 단계를 거쳐 $p_{\theta}(\mathbf{x})$를 샘플링할 수 있다.
- $x_{1} \sim p_{\theta}(x_{1} \mid x_{1:0})$을 샘플링한다.
- $x_{2} \sim p_{\theta}(x_{2} \mid x_{1})$을 샘플링한다.
- $x_{3} \sim p_{\theta}(x_{3} \mid x_{1:2})$를 샘플링한다.
- $\cdots$
- $x_{L} \sim p_{\theta}(x_{L} \mid x_{1:L-1})$을 샘플링한다.

$p_{\theta}(\mathbf{x})$가 정규화 조건을 만족하는 이유는 위의 샘플링 과정으로부터 바로 알 수 있다. $x_{1}$, $\cdots$, $x_{L}$을 모두 올바른 확률 분포에서 샘플링했기 때문이다. 이들을 합쳐 만든 $\mathbf{x}$ 또한 당연히 올바른 확률 분포를 따르는 확률 변수가 된다.

# Autoregressive Model의 학습

이제 autoregressive model을 어떻게 학습시키는지 알아보기 위해, $p_{\mathrm{data}}$와 $p_{\theta}$의 KL divergence를 최소화하는 문제를 생각해 보자. [이전 포스트](../01-overview)에서 보인 것처럼 이는 expected log-likelihood를 최대화하는 것, 즉 negative log-likelihood를 최소화하는 것과 동치이다. 따라서 목적 함수를 다음과 같이 정의하자.

$$
\begin{align*}
\theta &= \argmin_{\theta} J(\theta)\\
J(\theta) &= -\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} [\log p_{\theta}(\mathbf{x})]
\end{align*}
$$

식 {{< eqref p_theta-conditional >}}을 이용해 $J(\theta)$를 정리하면 다음과 같다.

$$
\begin{align*}
J(\theta) &= -\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \left[\log \prod_{i=1}^{L} p_{\theta}(x_{i} \mid x_{1:i-1})\right]\\
&= -\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \left[ \sum_{i=1}^{L} \log p_{\theta}(x_{i} \mid x_{1:i-1})\right]
\end{align*}
$$

$p_{\theta}(x_{i} \mid x_{1:i-1})$을 모두 단순한 분포로 설정했으므로, 이들의 밀도 함수나 $\theta$에 대한 gradient를 쉽게 구할 수 있다. 이제 목적 함수 $J(\theta)$의 gradient를 계산할 수 있는지 살펴보자.

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= -\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \left[ \sum_{i=1}^{L} \nabla_{\theta} \log p_{\theta}(x_{i} \mid x_{1:i-1})\right]
\end{align*}
$$

이 식은 몬테 카를로 근사를 적용할 수 있는 형태이다. 구체적으로는, $\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$이 $p_{\mathrm{data}}$에서 얻은 IID 샘플이라 하면 다음과 같이 근사할 수 있다. 아래첨자와 헷갈리지 않기 위해 $n$번째 샘플을 위첨자 $\mathbf{x}^{(n)}$로 표기했고, $\mathbf{x}^{(n)} = (x^{(n)}_{1}, \cdots, x^{(n)}_{L})$ 이다.

{{< eqlabel monte-carlo-gradient >}}
$$
\nabla_{\theta} J(\theta) \approx -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{L} \nabla_{\theta} \log p_{\theta}(x_{i}^{(n)} \mid x_{1:i-1}^{(n)})
$$

위 식은 계산 가능하므로, 우리는 $J(\theta)$에 대한 최적화 문제를 경사 하강법으로 풀 수 있다. 위 식에서는 데이터의 길이가 $L$로 고정되어 있지만, 데이터의 길이가 고정되지 않은 경우에 대해서도 마찬가지로 할 수 있다.

# Autoregressive Models on Discrete Tokens

이제 autoregressive model이 활용되는 대표적인 사례인 텍스트 생성에 대해 알아보자. 이미 [지난 포스트](../01-overview/#idea-1-autoregressive-generation)에서 설명했던 내용인데, 학습을 위한 목적 함수까지 포함해 다시 살펴보자.

텍스트를 토큰(token) 단위로 분해하면, 각 토큰은 $K$개의 고정된 vocabulary 중 하나이다 ($K$는 보통 수만에서 수십만 정도이다). $i$번째 토큰을 $x_{i}$로 놓으면, $p_{\theta}(x_{i} \mid x_{1:i-1})$는 $K$개의 값 중 하나를 선택하는 범주형 분포가 된다.

이 범주형 분포를 구체적으로 모델링해 보자. $f_{\theta}(x_{1:i-1})$을 이전 토큰들을 입력으로 받아 $K$차원의 실수 벡터를 출력하는 신경망이라 하고, 출력 벡터의 $k$번째 원소를 $f_{\theta}(x_{1:i-1})_{k}$로 표기하자. 이들을 logit이라 한다. '0 이상' 조건과 정규화 조건을 만족시키기 위해 logit에 [softmax 함수](https://en.wikipedia.org/wiki/Softmax_function)를 적용하면 다음과 같이 범주형 분포의 확률 분포를 얻을 수 있다.

{{< eqlabel softmax-categorical >}}
$$p_{\theta}(x_{i} = k \mid x_{1:i-1}) = \frac{\exp(f_{\theta}(x_{1:i-1})_{k})}{\sum_{j=1}^{K} \exp(f_{\theta}(x_{1:i-1})_{j})}$$

이제 앞 절의 목적 함수의 gradient (식 {{< eqref monte-carlo-gradient >}})에 이 조건부 분포를 대입해 보자. 이를 위해서는 다음과 같이 하나의 샘플 $\mathbf{x}^{(n)}$에 대한 목적 함수의 gradient를 구해야 한다.

$$
-\sum_{i=1}^{L} \nabla_{\theta} \log p_{\theta}(x_{i}^{(n)} \mid x_{1:i-1}^{(n)})
$$

식 {{< eqref softmax-categorical >}}을 대입하면, $i$번째 항은 다음과 같다.

$$-\nabla_{\theta} \log p_{\theta}(x^{(n)}_{i} \mid x^{(n)}_{1:i-1}) = -\nabla_{\theta}  f_{\theta}(x^{(n)}_{1:i-1})_{x^{(n)}_{i}} + \nabla_{\theta} \log \sum_{j=1}^{K} \exp(f_{\theta}(x^{(n)}_{1:i-1})_{j})$$

첨자가 많아 헷갈리기는 하지만, 아무튼 위 식은 쉽게 계산할 수 있는 식이다.

{{< callout type="Note" >}}
위에서는 목적 함수 $J(\theta)$의 gradient에 대한 식을 쓴 다음 몬테 카를로 근사를 적용했다. 학습을 위해 필요한 값이 $J(\theta)$가 아니라 $\nabla_{\theta} J(\theta)$이기 때문에 그렇게 한 것이다.

이번에는 목적 함수의 의미를 알아보기 위해, 목적 함수에 바로 몬테 카를로 근사를 적용해 $J(\theta)$를 계산 가능한 형태로 정리해 보자.

$$
\begin{align*}
J(\theta) &= -\mathbb{E}_{\mathrm{x} \sim p_{\mathrm{data}}} \left[ \sum_{i=1}^{L}\log p_{\theta}(x_{i} \mid x_{1:i-1})\right]\\
&\approx \frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{L} \left[ -\log p_{\theta}(x^{(n)}_{i} \mid x^{(n)}_{1:i-1}) \right]
\end{align*}
$$

하나의 샘플 $\mathbf{x}^{(n)}$만 고려하자. 식 {{< eqref softmax-categorical >}}을 대입하면 $i$번째 항은 다음과 같다.

$$-\log p_{\theta}(x^{(n)}_{i} \mid x^{(n)}_{1:i-1}) = -f_{\theta}(x^{(n)}_{1:i-1})_{x^{(n)}_{i}} + \log \sum_{j=1}^{K} \exp(f_{\theta}(x^{(n)}_{1:i-1})_{j})$$

이것은 기계 학습에서 흔히 사용하는 **cross-entropy loss** 와 정확히 동일한 형태이다. 즉, KL divergence 최소화를 이용한 autoregressive model의 학습은 각 위치에서 다음 토큰을 예측하는 $K$-클래스 분류 문제의 cross-entropy loss를 최소화하는 것과 동일하다.

{{< /callout >}}

학습 과정에서 주목할 점이 하나 있다. 샘플링 과정에서는 $x_{1}$부터 $x_{L}$까지 순차적으로 생성해야 하지만, 학습 과정에서는 정답 토큰 $x_{1}^{(n)}$, $\cdots$, $x_{L}^{(n)}$이 이미 주어져 있다. 따라서 $f_{\theta}(x_{1:i-1}^{(n)})$를 모든 $i$에 대해 동시에 계산할 수 있다. 이처럼 학습 시 정답 데이터를 신경망의 입력으로 사용하는 방식을 **teacher forcing**이라 한다. Teacher forcing 덕분에 학습은 효율적으로 병렬화할 수 있지만, 샘플링(생성)은 여전히 순차적으로 수행해야 한다.

Teacher forcing은 꼭 텍스트 생성과 같이 이산적인 분포에만 적용할 수 있는 것은 아니다. 하지만 여기에서 설명한 이유는 학습 과정을 구체적으로 살펴보아야 설명하기 편하기 때문이다.

# Limitations of Autoregressive Models

Autoregressive model의 한계를 간단히 살펴보자.

먼저 **순차적 샘플링**이 있다. 샘플링을 할 때 $x_{1}$부터 $x_{L}$까지 반드시 순서대로 생성해야 하므로, 병렬화가 불가능하고 생성 속도가 느리다. 이는 autoregressive model가 가진 가장 근본적인 한계이다.

다음으로, **Exposure bias**가 있다. 학습 시에는 정답 토큰을 입력으로 사용하는 teacher forcing을 통해 학습하지만, 샘플링 시에는 모델 자신의 출력을 입력으로 사용하기 때문이다. 이 불일치로 인해 샘플링 과정에서 오류가 누적될 수 있다.

# Autoregressive Models in Practice

이 포스트에서 다룬 내용이 autoregressive model의 핵심이다. 오늘날 수백억 개의 매개변수를 가진 대규모 언어 모델(LLM)도 동일한 autoregressive 구조 위에서 동작한다. 확률 분포의 모델링 자체는 이것으로 완성되었고, 실제로 강력한 모델을 만들기 위해서는 다음과 같은 문제들이 남아 있다. 지금은 수학적 모델링에 초점을 맞추고 있기 때문에 아래 문제들은 자세히 언급하지 않겠다.

- **신경망 구조**: $f_{\theta}$를 어떤 신경망으로 구현할 것인가?
  - 현재는 [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))가 사실상 표준으로 자리잡았다.
- **데이터**: 대규모 학습 데이터를 어떻게 수집하고 정제할 것인가?
- **학습과 샘플링의 효율성**: 학습과 샘플링 속도를 어떻게 개선할 것인가?
- **강화 학습**: 특정 목적을 더 잘 달성하는 데이터를 생성하려면 어떻게 해야 하는가?