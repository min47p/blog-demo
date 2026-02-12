---
title: "Generative Models - 03. Variational Autoencoder (1)"
date: 2026-02-13
tags: ["machine-learning", "generative-models"]
draft: false
---

잠재 변수(latent variable)을 사용하는 대표적인 모델인 variational autoencoder (VAE)에 대해 알아보자. VAE는 2013년 D. P. Kingma와 M. Welling이 개발했다{{< ref 1 >}}. D. P. Kingma는 Adam optimizer를 개발하기도 한 능력자이다{{< ref 2 >}}.

VAE가 만들어진 맥락을 이해하기 위해, 확률 모델에 대한 전통적인 추론 알고리즘에서부터 출발하려고 한다. 첫 포스트에서는 E-M algorithm, MCMC, variational Bayes에 대해 간단히 알아볼 것이다. 두 번째 포스트에서는 amortized inference에 대해 알아보고, wake-sleep 알고리즘을 살펴본 뒤 본격적으로 VAE에 대해 알아볼 것이다. 마지막 포스트에서는 Hierarchical VAE (HVAE)를 살펴볼 것이다.

# Latent Variable Models

먼저 이전 포스트에서 간단하게 살펴보았던 latent variable model에 대해 더 알아보자.

## 세 가지 확률 변수

지금부터 다룰 모델에는 세 가지 변수가 등장한다. 관측 데이터 $\mathbf{x}$, 잠재 변수 $\mathbf{z}$, 그리고 매개변수 $\theta$이다.

**관측 데이터(observed data)** 는 우리가 실제로 관측하고 수집할 수 있는 데이터이다. **잠재 변수(latent variable)** 나 **매개변수(parameter)** 는 우리가 관측할 수 없고, 존재한다고 가정하고 있는 값이다. 이 두 변수는 모델이 관측 데이터의 분포를 설명하기 위해 도입한 변수이며, 역할이 다르다. 잠재 변수는 각 관측 데이터마다 서로 다른 값을 가질 수 있다. 반면, 모든 관측 데이터는 같은 매개변수를 공유한다.

Claude가 만들어 준 다음 예시를 살펴보자. 매일 파스타를 만드는 요리사가 있다. 손님들은 파스타를 먹고 1점에서 10점까지 점수를 매긴다. 그런데 이 요리사는 매일 그 날의 기분에 따라 요리의 질이 달라진다. 요리사의 기분은 좋음과 나쁨 두 가지 상태가 있는데, 기분이 좋은 날에는 더 맛있는 요리를 만든다. 요리사의 기분을 직접 물어볼 수는 없다. 현실에서는 어제의 기분이 오늘의 기분에 영향을 주겠지만, 여기에서는 그런 요소는 생각하지 말고 각 날의 기분이 서로 독립이라고 가정하자.

여기에서 관측 데이터 $\mathbf{x}$는 특정 날짜에 손님들이 매긴 평점이다. $\mathbf{x}$의 분포를 모델링하기 위해 우리는 평점이 요리사의 기분과 관련이 있을 것이라 가정했는데, 이것이 잠재 변수 $\mathbf{z}$에 해당한다. 즉, $\mathbf{z}$는 '기분 좋음'과 '기분 나쁨' 두 가지 값을 가진다. 하지만 이것만으로 $\mathbf{x}$의 분포를 모두 설명할 수는 없다. 먼저 요리사가 어떤 날짜에 기분이 좋을 확률과 나쁠 확률이 각각 얼마인지 알아야 한다. 또한 요리사가 기분이 좋을 때와 나쁠 때 각각 손님들의 점수가 어떤 분포를 따르는지 알아야 한다. 이러한 것들을 결정하는 값들이 매개변수 $\theta$에 해당한다. 잠재 변수 $\mathbf{z}$는 매일 바뀌지만, $\theta$는 이 상황 전체에 대해 하나의 값으로 고정되어 있다.

예를 들어, 요리사가 어떤 날짜에 기분이 좋을 확률이 $\pi$, 나쁠 확률이 $1 - \pi$라고 정의하자. 그리고 기분이 좋을 때 손님들의 점수가 정규분포 $\mathcal{N}(\mu_{1}, \sigma^{2}_{1})$, 나쁠 때의 점수가 정규분포 $\mathcal{N}(\mu_{2}, \sigma^{2}_{2})$를 따른다고 모델링하자. 이때 매개변수는 $\theta = (\pi, \mu_{1}, \sigma_{1}^{2}, \mu_{2}, \sigma_{2}^{2})$이다.

이 상황에서, 우리가 궁금해할 수 있는 문제가 몇 가지 있다.
1. 매개변수 $\theta$ 구하기: 요리사의 평균 실력, 기분이 좋을 확률, 기분에 얼마나 영향을 받는지 등을 알고 싶다.
2. 잠재 변수 $\mathbf{z}$ 구하기: 특정 날의 점수를 보고, 요리사의 기분을 알고 싶다.
3. 관측 데이터 $\mathbf{x}$ 생성하기: 이 요리사가 내일 만들 파스타의 평점으로 그럴듯한 값을 생성하고 싶다.

우리가 관심 있는 목표는 3번 문제이지만, 1번과 2번 문제도 모두 연관되어 있다. 우리는 매개변수와 잠재 변수를 도입해 데이터를 설명하는 모델을 사용하고 있기 때문이다. 특히 이 포스트의 후반부에서는 2번 문제에 더 초점을 맞출 것이다.

## 생성 모델의 가정과 목표

이제 3번 문제를 풀고자 할 때, (1) 무엇이 가정이고, (2) 무엇이 주어져 있고, (3) 무엇이 목표인지 명확하게 써 보자. 이미 첫 포스트에서 다루었지만, latent variable model의 관점에서 다시 살펴보려고 한다.

먼저 (1) latent variable model의 가정은 다음과 같다.

- 매개변수 $\theta$가 주어졌을 때, 우리는 잠재 변수 $\mathbf{z}$의 분포를 알고 있다. 이 분포는 $p(\mathbf{z} \mid \theta)$ 또는 더 간단하게 $p_{\theta}(\mathbf{z})$로 표현할 수 있다.
  - 위의 예시에서 $p_{\theta}(\mathbf{z} = \textrm{기분 좋음}) = \pi$이고, $p_{\theta}(\mathbf{z} = \textrm{기분 나쁨}) = 1 - \pi$ 였다.
- 매개변수 $\theta$와 잠재 변수 $\mathbf{z}$가 주어졌을 때, 우리는 관측 데이터 $\mathbf{x}$의 분포를 알고 있다. 이 분포는 $p(\mathbf{x} \mid \mathbf{z}, \theta)$ 또는 더 간단하게 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$로 나타낸다.
  - 위의 예시에서 $p_{\theta}(\mathbf{x} \mid \mathbf{z} = \textrm{기분 좋음}) = \mathcal{N}(\mathbf{x}; \mu_{1}, \sigma_{1}^{2})$였다.
  - 또한, $p_{\theta}(\mathbf{x} \mid \mathbf{z} = \textrm{기분 나쁨}) = \mathcal{N}(\mathbf{x}; \mu_{2}, \sigma_{2}^{2})$ 였다.

분포를 알고 있다는 것은 정규 분포, 범주형 분포, [베타 분포](https://en.wikipedia.org/wiki/Beta_distribution) 등 우리가 아는 분포들과 신경망 $f_{\theta}$ 등 우리가 아는 함수들로 나타낼 수 있다는 의미이다.

(2) 문제를 풀기 위해 주어진 것은 $\mathbf{x}$의 실제 분포 $p_{\mathrm{data}}(\mathbf{x})$로부터 IID 샘플링한 $N$개의 샘플 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$이다.

(3) 우리의 목표는 어떤 매개변수 $\theta$를 잘 정해서 우리의 모델에서의 관측 데이터의 분포 $p_{\theta}(\mathbf{x})$가 실제 데이터의 분포 $p_{\mathrm{data}}(\mathbf{x})$와 비슷해지도록 하는 것이다. 비슷하다의 기준으로 KL divergence를 채택하면, 다음 문제를 푸는 것이 된다.
$$
\theta = \argmin_{\theta} D_{KL}(p_{\mathrm{data}} \| p_{\theta})
$$

KL divergence의 최소화는 MLE와 같으므로, 다음과 같이 쓸 수 있다.
$$
\theta = \argmax_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}}[\log p_{\theta}(\mathbf{x})]
$$

기댓값 자체는 몬테 카를로 근사를 이용해 없앨 수 있다. 그런데 likelihood $\log p_{\theta}(\mathbf{x})$가 문제이다. 우리 모델의 정의상 이것을 알고 있는 값들로 표현하려면 아래와 같은 식을 써야 한다. 그래서 $\log p_{\theta}(\mathbf{x})$를 **marginal likelihood**라고 부른다.

$$\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \log \int p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z}) \, d\mathbf{z}$$

이 식에는 복잡한 적분이 들어있어 계산할 수 없다(intractable). 좀 더 자세히 살펴보자.

## Intractability of Marginal Likelihood

적분이 $\mathbf{z}$에 대한 기댓값의 형태로 나타낼 수 있으므로, 몬테 카를로 근사를 적용하면 되지 않나? 라고 생각할 수 있다. 실제로, 다음과 같은 표현이 가능하다.
$$\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z}) \, d\mathbf{z} = \log \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z})}[p_{\theta}(\mathbf{x} \mid \mathbf{z})]$$

실제로 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$도 계산 가능하므로, $p_{\theta}(\mathbf{z})$에서 $\mathbf{z}$를 샘플링해 적분값을 근사하려는 아이디어는 좋아 보인다. 하지만 이 방법이 실제로 통하지 않는 이유는 사전 분포 $p_{\theta}(\mathbf{z})$에서 샘플링한 $\mathbf{z}$가 주어진 $\mathbf{x}$를 잘 설명할 가능성이 매우 낮기 때문이다. $\mathbf{z}$가 고차원일 때, 사전 분포가 차지하는 공간은 매우 넓지만 특정 $\mathbf{x}$에 대해 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 유의미한 값을 가지는 $\mathbf{z}$의 영역은 극히 작다. 따라서 대부분의 샘플에서 $p_{\theta}(\mathbf{x} \mid \mathbf{z}) \approx 0$이 되어, 유의미한 근사값을 얻기 어렵다.

이는 [첫 포스트](../01-overview/#monte-carlo-approximation)에서 잠깐 언급했던, 기댓값 내부의 분산이 지나치게 커 몬테 카를로 근사를 실질적으로 사용할 수 없는 상황이다. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$는 아주 좁은 영역의 $\mathbf{z}$에서만 유의미한 값을 가지고, 나머지 영역에서의 값은 $0$에 수렴하기 때문에 분산이 크다.

위 논의는 '유의미한 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 가지는 $z$의 값이 중요하다' 라는 힌트를 주고 있다. 그리고 이러한 $\mathbf{z}$는 분포 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$, 즉 관측 데이터 $\mathbf{x}$가 주어졌을 때 잠재 변수 $\mathbf{z}$의 분포와 관련이 있다. 왜냐하면 다음과 같은 베이즈 정리 때문이다.
$$
p_{\theta}(\mathbf{x} \mid \mathbf{z}) = \frac{p_{\theta}(\mathbf{z} \mid \mathbf{x}) p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{z})}
$$

위 식에서, $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 의미 있는 값이려면 분자에 위치한 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$도 의미 있는 값이어야 한다는 사실을 알 수 있다. $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 **사후 분포(posterior distribution)** 라고 한다. $\mathbf{x}$가 주어지기 전 $\mathbf{z}$의 분포, 즉 $p_{\theta}(\mathbf{z})$는 **사전 분포(prior distribution)** 라고 한다.

그런데 문제는 대부분의 경우 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 구하기도 어렵다는 점이다. 위 식에서는 우리가 처음에 구하려고 했던 $p_{\theta}(\mathbf{x})$도 분자에 들어 있어, 순환 논리에 빠진 것 같다는 느낌을 준다. 아무튼 우리의 모델에서 posterior를 구할 수 있다면 문제를 해결할 수 있다. 아래와 같이 $\log p_{\theta}(\mathbf{x})$의 gradient를 계산 가능한 식으로 바꿀 수 있기 때문이다.
{{< eqlabel marginal-as-posterior >}}
$$
\begin{align*}
\nabla_{\theta} \log p_{\theta}(\mathbf{x}) &= \frac{\nabla_{\theta} p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{x})} = \frac{1}{p_{\theta}(\mathbf{x})} \nabla_{\theta} \left(\int p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}\right)\\
&= \frac{1}{p_{\theta}(\mathbf{x})} \int \nabla_{\theta} p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}\\
&= \frac{1}{p_{\theta}(\mathbf{x})} \int p_{\theta}(\mathbf{x}, \mathbf{z})\,\frac{\nabla_{\theta} p_{\theta}(\mathbf{x}, \mathbf{z})}{p_{\theta}(\mathbf{x}, \mathbf{z})}\, d\mathbf{z}\\
&= \frac{1}{p_{\theta}(\mathbf{x})} \int p_{\theta}(\mathbf{x}, \mathbf{z})\,\nabla_{\theta} \log p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}\\
&= \frac{1}{p_{\theta}(\mathbf{x})} \int p_{\theta}(\mathbf{z} \mid \mathbf{x}) p_{\theta}(\mathbf{x})\,\nabla_{\theta} \log p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}\\
&= \int p_{\theta}(\mathbf{z} \mid \mathbf{x})\,\nabla_{\theta} \log p_{\theta}(\mathbf{x}, \mathbf{z})\, d\mathbf{z}\\
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x})} \left[ \nabla_{\theta} \log p_{\theta}(\mathbf{x}, \mathbf{z}) \right]\\
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x})} \left[ \nabla_{\theta} \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \right ]
\end{align*}
$$

유도 과정이 상당히 복잡해 보이지만, 결국 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 어떻게든 꺼내 이 분포에 대한 기댓값으로 바꾸기 위한 작업이다. 세 번째와 네 번째 등호에서는 $\nabla_{\theta} p_{\theta}(\mathbf{x}, \mathbf{z})$에서 $p_{\theta}(\mathbf{x}, \mathbf{z})$를 꺼내기 위해 로그함수를 사용했는데, 이 log-derivative trick은 통계학에서 자주 사용되는 아름다운 기술이다.

## E-M Algorithm

Posterior를 구할 수 있을 때 **Expectation-Maximization algorithm (E-M)** 을 적용할 수 있다{{< ref 3 >}}. E-M algorithm은 신경망과 같이 복잡한 함수가 들어간 모델이 아직 사용되지 않던 20세기 후반에 등장했으며, 모델이 단순해 posterior를 해석적으로 구할 수 있을 때 강력한 성능을 발휘한다. 우리 맥락에서는 없어도 되지만, 매우 유명한 알고리즘이므로 잠깐 언급하려고 한다.

E-M algorithm은 $\mathbf{z}$의 posterior distribution을 구하는 **E-step**과 $\theta$를 최적화하는 **M-step**을 번갈아가며 실행하는 알고리즘이다. 구체적인 방법은 다음과 같다.

* 샘플 $\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$이 주어져 있다 하자. 매개변수의 초기값 $\theta_{0}$을 잡자.
* 다음 두 step을 충분히 반복한다. $t$번째 반복에서는 다음을 수행한다. ($t = 1, 2, \cdots$)
  * **E-step** (expectation step): 매개변수를 $\theta_{t-1}$로 고정하고, 각 샘플에 대한 posterior를 구한다. 즉, $n = 1, 2, \cdots, N$에 대해 $p^{(n)}_{t}(\mathbf{z}) = p_{\theta_{t - 1}}(\mathbf{z} \mid \mathbf{x} = \mathbf{x}^{(n)})$를 구한다.
  * **M-step** (maximization step): E-step에서 구한 posterior들을 이용해, 매개변수에 대한 최적화 문제를 풀어 $\theta_{t}$를 구한다.

M-step에 대한 설명이 더 필요하다. 매개변수를 최적화한다는 것은 아래 MLE 문제를 푼다는 것이다.
$$
\begin{align*}
\theta_{t} &= \argmax_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [ \log p_{\theta}(\mathbf{x})]\\
&\approx \argmax_{\theta} \sum_{n = 1}^{N} \log p_{\theta}(\mathbf{x}^{(n)})\\
\end{align*}$$

여기에서 $\log p_{\theta}(\mathbf{x})$는 다음과 같이 posterior distribution으로 나타낼 수 있다.
{{< eqlabel maximization-without-e-step >}}
$$
\begin{align*}
\log p_{\theta}(\mathbf{x}^{(n)}) &= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)})} [\log p_{\theta}(\mathbf{x}^{(n)})]\\
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)})} \left[ \log \frac{p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z})}{p_{\theta}(\mathbf{z \mid x^{(n)}})}\right]\\
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)})}[\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}) - \log p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)})]
\end{align*}
$$
위 식의 첫 번째 등호에서, $\mathbf{z}$와 직접적으로 연관이 없는 식인 $\log p_{\theta}(\mathbf{x})$를 posterior와 연관시키기 위해 강제로 기댓값을 씌우는 트릭을 사용했다.

위 식만 보면 이걸 최대화하는 $\theta$를 찾기 난감한 면이 있다. E-M algorithm의 아이디어는 E-step에서 구한 posterior distribution을 활용하는 것이다. 식 {{< eqref maximization-without-e-step >}}에 E-step에서 구한 분포를 대입하면 다음과 같이 쓸 수 있다.

$$
\log p_{\theta}(\mathbf{x}^{(n)}) \approx \mathbb{E}_{\mathbf{z} \sim p^{(n)}_{t}(\mathbf{z})} [\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}) - \log p_{t}^{(n)}(\mathbf{z})]
$$

여기에서 기댓값 안의 두 번째 항은 $\theta$와 관련이 없으므로 최적화 문제에서 생략할 수 있다.

{{< callout type="Note" >}}
위 식에서는 등호 대신 근사 기호 $\approx$를 사용했다. 등호가 성립하지 않는 이유는 최적화 변수 $\theta$ 중 일부를 (멋대로) $\theta_{t - 1}$로 고정했기 때문이다. $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)}) = p_{t}^{(n)}(\mathbf{z})$로 놓았는데, 정의상 $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(n)}) = p_{\theta_{t-1}}(\mathbf{z} \mid \mathbf{x}^{(n)})$로 설정한 것과 같다.

사실, 위 식에서 다음과 같은 부등호 관계가 성립한다.
$$
\log p_{\theta}(\mathbf{x}^{(n)}) \ge \mathbb{E}_{\mathbf{z} \sim p^{(n)}_{t}(\mathbf{z})} [\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}) - \log p_{t}^{(n)}(\mathbf{z})]
$$

이 식은 E-M algorithm의 정당성을 보장해 주는 중요한 식이다. 지금은 E-M algorithm을 엄밀하게 설명하고 있지 않고, 같은 형태의 식이 아래에서 variational Bayes의 **ELBO**를 설명할 때 다시 나오기 때문에 넘어가겠다.
{{< /callout >}}

결국 M-step은 다음을 만족하는 $\theta_{t}$를 찾는 문제가 된다.
$$
\begin{align*}
\theta_{t} &\approx \argmax_{\theta} \sum_{n = 1}^{N} \log p_{\theta}(\mathbf{x}^{(n)})\\
&\approx \argmax_{\theta} \sum_{n=1}^{N} \mathbb{E}_{\mathbf{z} \sim p^{(n)}_{t}(\mathbf{z})} [\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z})]
\end{align*}
$$

이 문제는 몬테 카를로 근사를 적용해 해결할 수 있다. 왜냐하면 이제 $\mathbf{z}$가 posterior distribution인 $p_{t}^{(n)}(\mathbf{z})$에서 샘플링되므로, 위에서 언급한 분산이 지나치게 커지는 문제가 발생하지 않기 때문이다. 사실 **Gaussian mixture model (GMM)** 과 같이 단순한 모델의 경우에는 몬테 카를로 근사를 적용할 필요도 없이 위 식의 값을 해석적으로 구할 수 있다. 이 경우에 E-M algorithm은 매우 유용하다.

물론 여기에서 논의가 끝나는 것은 아니다. E-step과 M-step을 반복해 최적의 매개변수를 찾을 수 있으려면 '$\theta_{t}$이 항상 $\theta_{t-1}$보다 더 좋은 매개변수인가?'와 같은 질문에 답할 수 있어야 한다. 여기에서는 생략한다.

마지막으로 E-step에서 posterior $p_{t}^{(n)}(\mathbf{z})$를 구하는 방법에 대해 잠깐 언급하겠다. GMM처럼 단순한 모델에서는 posterior를 해석적으로 구할 수 있다. 하지만 모델이 조금만 복잡해지더라도 posterior를 해석적으로 구할 수 있으리라 기대하기 어렵다. 이런 경우에는 바로 뒤에서 설명할 베이지안 추론 방법들을 이용해 posterior를 근사해 구할 수 있다.

{{< toggle title="GMM의 정의와 posterior (E-step)" >}}
**Gaussian Mixture Model (GMM)** 은 데이터가 $K$개의 정규 분포의 혼합으로 생성된다고 가정하는 모델이다. 잠재 변수 $\mathbf{z}$는 $K$개의 값 중 하나를 가지며, 어떤 정규 분포에서 데이터가 생성되었는지를 나타낸다. 앞의 파스타 요리사 예시는 $K = 2$인 GMM이다. 요리사의 기분이 좋을 때와 나쁠 때의 점수가 각각 정규분포를 따르고 있었기 때문이다.

GMM의 구성 요소는 다음과 같다.

- 사전 분포: $p_{\theta}(\mathbf{z} = k) = \pi_{k}$, 단 $\sum_{k=1}^{K} \pi_{k} = 1$
- 조건부 분포: $p_{\theta}(\mathbf{x} \mid \mathbf{z} = k) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k})$
- 매개변수: $\theta = \pi_{k}, \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}$

GMM에서 $\mathbf{z}$는 이산 변수이므로, marginal distribution은 적분이 아닌 유한한 합으로 표현된다.

$$p_{\theta}(\mathbf{x}) = \sum_{k=1}^{K} \pi_{k} \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k})$$

Posterior도 베이즈 정리를 적용하면 바로 구할 수 있다. 분모가 유한한 합이기 때문이다.

$$p_{\theta}(\mathbf{z} = k \mid \mathbf{x}) = \frac{\pi_{k} \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k})}{\sum_{j=1}^{K} \pi_{j} \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{j}, \boldsymbol{\Sigma}_{j})}$$

이처럼 $\mathbf{z}$가 이산적이고 값의 수가 적은 경우에는 posterior를 해석적으로 구할 수 있다.
{{< /toggle >}}

{{< toggle title="GMM의 M-step" >}}
E-step에서 구한 posterior를 $r_{nk} = p_{\theta_{t-1}}(\mathbf{z} = k \mid \mathbf{x}^{(n)})$로 표기하자. 이 값을 responsibility라 부른다. $r_{nk}$는 $n$번째 데이터가 $k$번째 정규 분포에서 생성되었을 확률을 나타낸다.

M-step의 목적 함수를 GMM에 대해 구체적으로 쓰면 다음과 같다.

$$
\begin{align*}
\sum_{n=1}^{N} \mathbb{E}_{\mathbf{z} \sim p_{t}^{(n)}(\mathbf{z})} [\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z})] = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \log \left[ \pi_{k} \, \mathcal{N}(\mathbf{x}^{(n)}; \boldsymbol{\mu}_{k}, \boldsymbol{\Sigma}_{k}) \right]
\end{align*}
$$

이 식을 각 매개변수에 대해 미분하고 0으로 놓으면, 닫힌 형태(closed form)의 해를 얻을 수 있다. 편의상 $N_{k} = \sum_{n=1}^{N} r_{nk}$로 정의하면 결과는 다음과 같다. 계산 과정은 생략한다.

$$
\begin{align*}
\pi_{k} &= \frac{N_{k}}{N}\\
\boldsymbol{\mu}_{k} &= \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk} \, \mathbf{x}^{(n)}\\
\boldsymbol{\Sigma}_{k} &= \frac{1}{N_{k}} \sum_{n=1}^{N} r_{nk} \, (\mathbf{x}^{(n)} - \boldsymbol{\mu}_{k})(\mathbf{x}^{(n)} - \boldsymbol{\mu}_{k})^{\top}
\end{align*}$$

따라서 몬테 카를로 근사 등의 방법을 사용하지 않고도 M-step을 수행할 수 있다.
{{< /toggle >}}

# Bayesian Inference

앞 절에서 posterior distribution $p_{\theta}(\mathbf{z} \mid \mathbf{x})$이 중요하다는 사실을 알게 되었다. Posterior에 대한 문제는 전통적으로 **베이지안 추론(Bayesian inference)** 이라는 분야에서 다루고 있다. 이 절에서는 베이지안 추론 방법들 중 MCMC와 variational Bayes를 살펴볼 것이다. VAE는 이 중 variational Bayes에서 강하게 영향을 받았다 (논문 제목에도 들어가 있다).

이 절에서 풀고 싶은 문제는 다음과 같다.

{{< callout type="Problem" >}}
Latent variable model에서, 하나의 관측 데이터 샘플 $\mathbf{x}$가 주어졌을 때 $\mathbf{z}$의 posterior distribution $p_{\theta}(\mathbf{z} \mid \mathbf{x})$을 어떻게 구하거나 근사할 수 있을까?
{{< /callout >}}

이제부터는 $\theta$의 최적화 문제에 신경쓰지 않고, $\theta$는 이미 정해져서 잘 알고 있는 값이라고 가정한다. 그리고 샘플 $\mathbf{x}$도 하나로 고정한다. 즉 $\mathbf{x}$와 $\theta$가 고정된 상태에서 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 잘 근사하는 것에만 초점을 맞출 것이다.

{{< callout type="Note" >}}
모든 불확실함을 어떤 확률분포를 따르는 확률 변수로 보는 관점을 베이즈주의라고 한다. 베이지안 추론은 이런 베이즈주의의 관점에서, 관측 가능한 변수를 바탕으로 관측 불가능한 변수의 분포를 알아내고자 하는 분야이다.

우리는 지금까지 최적의 매개변수 $\theta$의 값을 구하는 데 초점을 맞추고 있었다. 하지만 베이즈주의에 따르면 매개변수 $\theta$도 확률 변수이기 때문에, $\theta$의 분포, 즉 prior를 고려해야 한다. 즉, 우리의 논의는 엄밀히 말해서는 완전히 베이지안(fully Bayesian)이 아니다.

이후에 설명할 MCMC나 variational Bayes는 편의상 $\theta$의 prior를 고려하지 않았다. 하지만 이 방법들은 원래는 $\theta$의 prior도 고려하는 fully Bayesian approach이다. 또한, 잠재 변수 $\mathbf{z}$ 없이 관측 데이터와 매개변수만 있는 상황에서도 적용할 수 있는 방법들이다.
{{< /callout >}}

## Markov Chain Monte Carlo (MCMC)

수학이나 컴퓨터과학에서 결정론적(deterministic) 알고리즘으로 풀기 어려운 문제가 랜덤 알고리즘으로 풀리는 경우가 많다. 예를 들어, 지금까지 자주 등장한 몬테 카를로 근사, 딥러닝이라는 것 자체를 가능하게 해 주는 [stochastic gradient descent (SGD)](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), 그리고 소수 판별에서의 [Miller-Rabin algorithm](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test), 소인수분해에서의 [Pollard's $\rho$ algorithm](https://en.wikipedia.org/wiki/Pollard%27s_rho_algorithm)이 있다.

베이지안 추론에서의 **Markov chain Monte Carlo (MCMC)** 알고리즘도 비슷하게 랜덤 샘플링이라는 강력한 도구를 이용해 문제를 해결한다. 구체적으로는 미지의 확률 분포인 $\mathbf{z}$의 posterior에서 (근사적으로) 샘플링이 가능하도록 해 준다.

$p_{\theta}(\mathbf{z} \mid \mathbf{x})$에서의 샘플링이 왜 어려울까? 베이즈 정리를 다시 써 보자.
$$
p_{\theta}(\mathbf{z} \mid \mathbf{x}) = \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{p_{\theta}(\mathbf{x})}
$$

분자의 $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{x} \mid \mathbf{z}) p_{\theta}(\mathbf{z})$는 모델링으로부터 알 수 있는 값이다. 하지만 분모에 있는 $p_{\theta}(\mathbf{x})$가 문제이다. 이것은 다름 아닌 marginal likelihood이고, 우리는 애초에 저 값을 구할 수 없어서 posterior에 관심을 갖게 된 것이다. 그래서 저 값이 꼭 필요하다면 아무런 의미가 없는 순환 논리이다.

### Rejection Sampling

한편, 샘플 $\mathbf{x}$가 고정되어 있다면, 임의의 $\mathbf{z}$의 posterior의 밀도가 우리가 구할 수 있는 식에 **비례**한다는 것을 다음과 같이 알 수 있다.
$$
p_{\theta}(\mathbf{z} \mid \mathbf{x}) \propto  p_{\theta}(\mathbf{x},\mathbf{z}) = p_{\theta}(\mathbf{x} \mid \mathbf{z}) p_{\theta}(\mathbf{z})
$$

이것이 핵심이다. 이 식과 몬테 카를로 기법을 이용해 분모에 있는 $p_{\theta}(\mathbf{x})$를 몰라도 샘플링을 할 수 있다. 아이디어는 일단 $\mathbf{z}$를 우리가 아는 분포에서 샘플링한 뒤, 확률 밀도에 비례하게 채택하고 나머지는 기각하는 것이다. 구체적인 방법은 다음과 같다.

0. 충분히 큰 양의 실수 $M$을 잡는다. $M$은 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$의 최댓값보다 커야 한다.
1. $\mathbf{z}$를 $p_{\theta}(\mathbf{z})$에서 랜덤 샘플링한다.
2. 실수 $t$를 균등 분포 $\mathcal{U}(0, M)$에서 랜덤 샘플링한다.
3. 이제 1에서 샘플링한 $\mathbf{z}$를 바탕으로 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 계산한다. 이 값이 $t$ 이상이면 $\mathbf{z}$를 채택하고, 아니면 $\mathbf{z}$를 기각한 뒤 1로 돌아가 다시 샘플링한다.

이 방법을 **rejection sampling**이라고 한다. 직관적으로 보면 단계 1에서 $\mathbf{z}$가 $p_{\theta}(\mathbf{z})$에 비례하는 밀도로 샘플링되고, 단계 3에서 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$에 비례하는 확률로 채택되므로, 채택된 $\mathbf{z}$의 밀도는 $p_{\theta}(\mathbf{x}, \mathbf{z})$에 비례한다는 것을 알 수 있다. 구체적인 증명 과정은 다음과 같다.

{{< toggle title="Rejection sampling의 증명" >}}
채택된 $\mathbf{z}$의 분포를 구하기 위해 베이즈 정리를 적용하자. 먼저 주어진 $\mathbf{z}$가 채택될 확률을 계산한다. 단계 2에서 $t$는 $[0, M]$에서 균등하게 샘플링되므로, $t \leq p_{\theta}(\mathbf{x} \mid \mathbf{z})$일 확률은

$$
P(\text{채택} \mid \mathbf{z}) = \frac{p_{\theta}(\mathbf{x} \mid \mathbf{z})}{M}
$$

이다. 이로부터 전체 채택 확률을 구할 수 있다.

$$
P(\text{채택}) = \int P(\text{채택} \mid \mathbf{z}') \, p_{\theta}(\mathbf{z}') \, d\mathbf{z}' = \frac{1}{M} \int p_{\theta}(\mathbf{x} \mid \mathbf{z}') \, p_{\theta}(\mathbf{z}') \, d\mathbf{z}' = \frac{p_{\theta}(\mathbf{x})}{M}
$$

따라서 베이즈 정리에 의해

$$
p(\mathbf{z} \mid \text{채택}) = \frac{P(\text{채택} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z})}{P(\text{채택})} = \frac{p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z}) / M}{p_{\theta}(\mathbf{x}) / M} = \frac{p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z})}{p_{\theta}(\mathbf{x})} = p_{\theta}(\mathbf{z} \mid \mathbf{x})
$$

이므로, 채택된 $\mathbf{z}$는 정확히 posterior distribution $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 따른다.
{{< /toggle >}}

아쉽게도 이 신기한 아이디어만으로는 부족하다. 앞에서와 비슷한 이유인데, 대부분의 $\mathbf{z}$에서는 $p_{\theta}(\mathbf{x} \mid \mathbf{z}) \approx 0$이기 때문에 단계 3에서 $\mathbf{z}$가 거의 항상 기각된다. $\mathbf{z}$가 채택되기 위해서는 매우 많은 반복을 거쳐야 하기 때문에, rejection sampling은 실제로 활용하지 못할 만큼 비효율적이다. 하지만 원점으로 돌아간 것은 아니다. 이 아이디어와 Markov chain을 접목시켜 샘플링 속도를 높일 수 있기 때문이다. 이러한 방법을 Markov chain Monte Carlo (MCMC)라고 한다. 대표적인 방법인 **Metropolis-Hastings algorithm**을 살펴보자.

### Metropolis-Hastings Algorithm

Metropolis-Hastings (MH) algorithm의 핵심 아이디어는 rejection sampling처럼 후보를 제안하고 채택/기각하되, 매번 사전 분포에서 독립적으로 샘플링하는 대신 현재 상태에서 가까운 후보를 제안하는 것이다. 이를 위해 우리가 자유롭게 선택할 수 있는 **proposal distribution** $q(\mathbf{z}' \mid \mathbf{z})$를 사용한다. 이 분포는 현재 상태 $\mathbf{z}$가 주어졌을 때 다음 후보 $\mathbf{z}'$를 제안하는 역할을 한다.

Proposal distribution으로는 우리가 원하는 분포를 사용할 수 있는데, 정규분포 $q(\mathbf{z}' \mid \mathbf{z}) = \mathcal{N}(\mathbf{z}, \sigma^{2} I)$가 대표적이다. 이렇게 현재 있는 샘플을 기반으로 다음 샘플을 만드는 것이 핵심 아이디어이며, 덕분에 rejection sampling보다 우리가 원하는 분포를 빠르게 만들어낼 수 있다.

구체적인 알고리즘은 다음과 같다. $\mathbf{z}^{(0)}$을 임의의 초깃값으로 설정한 뒤, 매 단계 $t$에서 다음을 반복한다. ($t = 1, 2, \cdots$)

1. $\mathbf{z}^{(t-1)}$가 주어진 제안 분포에서 $\mathbf{z}^{(t)}$의 후보 $\mathbf{z}'$를 샘플링한다. 즉, $\mathbf{z}' \sim q(\mathbf{z}' \mid \mathbf{z}^{(t-1)})$
2. 채택 확률 $\alpha$를 다음과 같이 계산한다:
$$
\alpha = \min \left( 1, \; \frac{p_{\theta}(\mathbf{x}, \mathbf{z}')}{p_{\theta}(\mathbf{x}, \mathbf{z}^{(t-1)})} \cdot \frac{q(\mathbf{z}^{(t-1)} \mid \mathbf{z}')}{q(\mathbf{z}' \mid \mathbf{z}^{(t-1)})} \right)
$$
3. 확률 $\alpha$로 $\mathbf{z}'$를 채택해 $\mathbf{z}^{(t)} = \mathbf{z}'$로 설정하고, 확률 $1 - \alpha$로 $\mathbf{z}'$를 기각해 $\mathbf{z}^{(t)} = \mathbf{z}^{(t-1)}$로 설정한다.

이때 $t$가 충분히 커지면 $\mathbf{z}^{(t)}$의 분포가 posterior인 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$로 수렴한다는 것을 증명할 수 있다. 더 정확하게 표현하면, 위 Markov process의 stationary distribution이 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$이고, 적절한 조건 하에서 $\mathbf{z}^{(t)}$의 분포는 stationary distribution에 수렴하게 된다. 여기에서는 stationary distribution이 posterior라는 것만 확인하자.

{{< toggle title="MH algorithm의 stationary distribution" >}}
$\pi(\mathbf{z}) = p_{\theta}(\mathbf{z} \mid \mathbf{x})$가 위 Markov chain의 stationary distribution임을 보이려면, **detailed balance** 조건을 확인하면 된다. Detailed balance 조건이란 다음을 말한다. 아래 식에서 $T(\mathbf{z} \to \mathbf{z}')$는 Markov chain의 **전이 확률(transition probability)** 로, 현재 상태가 $\mathbf{z}$일 때 다음 상태가 $\mathbf{z}'$일 확률 밀도를 나타낸다.

$$
\pi(\mathbf{z}) \, T(\mathbf{z} \to \mathbf{z}') = \pi(\mathbf{z}') \, T(\mathbf{z}' \to \mathbf{z})
$$

$\pi$가 **stationary distribution** 이라는 것은, 현재 상태가 $\pi$를 따를 때 다음 상태도 $\pi$를 따르게 된다는 의미이다. 그렇게 되기 위해서는 $\pi(\mathbf{z}') = \int \pi(\mathbf{z}) \, T(\mathbf{z} \to \mathbf{z}') \, d\mathbf{z}$를 만족해야 한다. Detailed balance가 성립하면 양변을 $\mathbf{z}$에 대해 적분했을 때 이 조건이 만족되므로, $\pi$가 stationary distribution이라는 것을 바로 알 수 있다.

MH algorithm에서 전이 확률 $T(\mathbf{z} \to \mathbf{z}')$를 구해 보자. 상태 $\mathbf{z}$에서 $\mathbf{z}'$로 전이하려면, 먼저 제안 분포 $q(\mathbf{z}' \mid \mathbf{z})$에서 $\mathbf{z}'$가 제안되어야 하고, 그 후 확률 $\alpha(\mathbf{z}, \mathbf{z}')$로 채택되어야 한다. 따라서 $\mathbf{z} \neq \mathbf{z}'$일 때 전이 확률은 다음과 같다.
$$
T(\mathbf{z} \to \mathbf{z}') = q(\mathbf{z}' \mid \mathbf{z}) \cdot \alpha(\mathbf{z}, \mathbf{z}')
$$

($\mathbf{z}' = \mathbf{z}$인 경우, 즉 제자리에 머무는 확률은 후보가 기각될 확률이다. 이 경우는 detailed balance 증명에 필요하지 않으므로 넘어간다.)

여기에서 $\alpha(\mathbf{z}, \mathbf{z}')$는 채택 확률이다. 알고리즘의 정의에서 $p_{\theta}(\mathbf{x}, \mathbf{z})$의 비를 사용했는데, 비를 취할 때 $p_{\theta}(\mathbf{x})$가 약분되므로 이는 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x}) = \pi(\mathbf{\mathbf{z}})$의 비와 같다. 따라서,

$$
\alpha(\mathbf{z}, \mathbf{z}') = \min \left( 1, \; \frac{\pi(\mathbf{z}')}{\pi(\mathbf{z})} \cdot \frac{q(\mathbf{z} \mid \mathbf{z}')}{q(\mathbf{z}' \mid \mathbf{z})} \right)
$$

일반성을 잃지 않고 $\pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z}) \leq \pi(\mathbf{z}') \, q(\mathbf{z} \mid \mathbf{z}')$라고 가정하자. (Detailed balance 조건의 양변에서 $\mathbf{z}$와 $\mathbf{z}'$의 역할이 대칭적이므로, 반대의 경우는 $\mathbf{z}$와 $\mathbf{z}'$를 바꾸면 된다.) 그러면 $\alpha(\mathbf{z}, \mathbf{z}') = 1$이고,

$$
\alpha(\mathbf{z}', \mathbf{z}) = \frac{\pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z})}{\pi(\mathbf{z}') \, q(\mathbf{z} \mid \mathbf{z}')}
$$

이다. 이제 detailed balance를 확인하자.

$$
\begin{align*}
\pi(\mathbf{z}) \, T(\mathbf{z} \to \mathbf{z}') &= \pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z}) \cdot 1 = \pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z})\\
\pi(\mathbf{z}') \, T(\mathbf{z}' \to \mathbf{z}) &= \pi(\mathbf{z}') \, q(\mathbf{z} \mid \mathbf{z}') \cdot \frac{\pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z})}{\pi(\mathbf{z}') \, q(\mathbf{z} \mid \mathbf{z}')} = \pi(\mathbf{z}) \, q(\mathbf{z}' \mid \mathbf{z})
\end{align*}
$$

두 값이 같으므로 detailed balance가 성립한다. 따라서 $\pi(\mathbf{z}) = p_{\theta}(\mathbf{z} \mid \mathbf{x})$는 이 Markov chain의 stationary distribution이다.
{{< /toggle >}}

이 알고리즘의 핵심은 채택 확률을 계산할 때 $p_{\theta}(\mathbf{x}, \mathbf{z})$의 **비율**만 중요하다는 것이다. 비율을 계산할 때 정규화 상수 $p_{\theta}(\mathbf{x})$가 약분되기 때문에, posterior를 정규화할 수 없어도 알고리즘을 실행할 수 있다. 제안 분포가 대칭적인 경우, 즉 $q(\mathbf{z}' \mid \mathbf{z}) = q(\mathbf{z} \mid \mathbf{z}')$이면 $\alpha$의 두 번째 항이 1이 되어 식이 더 간단해진다. 예를 들어, $q(\mathbf{z}' \mid \mathbf{z}) = \mathcal{N}(\mathbf{z}, \sigma^{2} I)$처럼 현재 위치를 중심으로 한 정규분포를 사용하면 된다.

### Limitations of MCMC

MCMC는 강력한 방법이지만 몇 가지 한계가 있다. 가장 큰 문제는 수렴 속도이다. MCMC는 rejection sampling보다는 효율적이지만, 여전히 posterior에 수렴하기까지는 충분한 반복이 필요하다. 그리고 잠재 변수의 차원이 높아지면 이 시간이 급격히 늘어난다. 딥러닝 모델에서는 $\mathbf{z}$의 차원이 수백에서 수천에 이르기 때문에, 하나의 $\mathbf{x}$에 대해서만 posterior 샘플을 얻는 데도 매우 오래 걸린다.

그래서 MCMC를 학습에 적용하기 쉽지 않다. 우리가 하고 싶은 것은 식 {{< eqref marginal-as-posterior >}}을 계산하는 것인데, 여기로 옮겨 온 뒤 몬테 카를로 근사까지 적용해서 다시 써 보자. 먼저 $K$개의 샘플 $\mathbf{z}^{(1)}$, $\cdots$, $\mathbf{z}^{(K)}$를 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$에서 MCMC로 샘플링하면 식 {{< eqref marginal-as-posterior >}}을 다음과 같이 근사할 수 있다.
$$
\begin{align*}
\nabla_{\theta} \log p_{\theta}(\mathbf{x})
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x})} \left[ \nabla_{\theta} \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \nabla_{\theta} \log p_{\theta}(\mathbf{z}) \right ]\\
&\approx \frac{1}{K} \sum_{k = 1}^{K} \left( \nabla_{\theta} \log p_{\theta}(\mathbf{x} \mid \mathbf{z}^{(k)}) + \nabla_{\theta} \log p_{\theta}(\mathbf{z}^{(k)}) \right)
\end{align*}
$$

이제 모든 샘플 $\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$에 대한 위 값의 평균이 $\theta$의 gradient의 몬테 카를로 근사값이므로, 

$$
\begin{align*}
\nabla_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}} [\log p_{\theta}(\mathbf{x})]
&\approx \frac{1}{N} \frac{1}{K} \sum_{n=1}^{N} \sum_{k = 1}^{K} \left( \nabla_{\theta} \log p_{\theta}(\mathbf{x}^{(n)} \mid \mathbf{z}^{(n, k)}) + \nabla_{\theta} \log p_{\theta}(\mathbf{z}^{(n, k)}) \right)
\end{align*}
$$

경사 하강법으로 매개변수 $\theta$를 최적화하려면 매 step마다 위 식을 여러 번 계산해야 한다. 위 식에서 $\mathbf{z}$의 샘플이 $NK$개 필요하다는 점에 주목하자. $\theta$가 업데이트될 때마다 posterior가 바뀌므로, 매번 $NK$개의 샘플을 새로 뽑아야 한다. 그런데 각 샘플은 Markov chain에서 많은 step을 거쳐야 뽑을 수 있다. 이런 사중 반복 구조 때문에 학습 전체가 극도로 느려진다.

결과적으로 MCMC는 생성 모델을 학습하는 데 활용하기에 적합하지 않다. MCMC는 energy-based model을 설명할 때 다시 등장하여, 학습이 아니라 학습이 완료된 모델에서 샘플링하는 데 활용될 것이다. 이제 다음 방법인 variational Bayes로 넘어가자.

## Variational Bayes

Posterior를 구하는 문제를 다른 관점으로 접근해 보자. 우리는 생성 모델을 학습시키기 위해 실제 데이터의 분포 $p_{\mathrm{data}}(\mathbf{x})$를 매개화된 확률 분포 $p_{\theta}(\mathbf{x})$로 근사하고 있다. 이제 우리는 하나의 샘플 $\mathbf{x}$가 주어졌을 때 posterior인 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 알고 싶은데, 비슷하게 '새로운 매개변수 $\phi$와 매개화된 확률 분포 $q_{\phi}(\mathbf{z})$를 도입해 $q_{\phi}(\mathbf{z}) \approx p_{\theta}(\mathbf{z} \mid \mathbf{x})$로 근사하자' 라는 아이디어를 활용할 수 있다. 이러한 접근 방법을 **variational Bayes**라고 한다. 'Variational'이라는 단어는 확률 변수의 분산(variation)과는 별 관련이 없고, 수학에서 최적의 함수를 찾는 분야인 변분법(calculus of variations)이나 변분 해석(variational analysis)에서 왔다.

지금까지 했던 대로, $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 $q_{\phi}(\mathbf{z})$로 근사하는 것을 두 분포 사이의 KL divergence를 최소화하는 문제로 모델링하자. 그런데 여기서 생각해 봐야 하는 점이, 지금까지는 KL divergence에서 미지의 분포를 왼쪽에, 모델링하는 분포를 오른쪽에 두었다. 그런데 KL divergence는 비대칭적이기 때문에 이렇게 하는 것이 당연하지는 않다. 지금까지 미지의 분포를 왼쪽에 둔 이유는 [첫 포스트](../01-overview/#monte-carlo-approximation)에서 살펴보았듯이 미지의 분포를 오른쪽에 두면 몬테 카를로 근사가 불가능하기 때문이다. 그렇다면 이번에도 그럴까? 결론적으로 말하자면, 이번에는 반대로 미지의 분포를 오른쪽에 두어야 계산 가능한 식이 나온다. 일단 이걸 모른다고 생각하고, 둘 다 해보자.

### 미지의 분포를 왼쪽에 두기

{{< callout type="Problem" >}}
하나의 관측 데이터 샘플 $\mathbf{x}$가 주어졌을 때 다음 최적화 문제를 풀자.
$$\phi = \argmin_{\phi} D_{\mathrm{KL}} (p_{\theta}(\mathbf{z} \mid \mathbf{x}) \| q_{\phi}(\mathbf{z}))
$$
{{< /callout >}}

최적화 문제의 목적 함수를 $J_{1}(\phi)$라 하고, 우리가 계산할 수 있는 형태로 표현해 보자. $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 우리가 아는 값인 $p_{\theta}(\mathbf{x}, \mathbf{z})$로 바꾸는 것이 핵심이다.
$$
\begin{align*}
J_{1}(\phi) &= D_{\mathrm{KL}}(p_{\theta}(\mathbf{z} \mid \mathbf{x}) \| q_{\phi}(\mathbf{z}))\\
&= \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \mathbf{x})} \left[\log \frac{p_{\theta}(\mathbf{z} \mid \mathbf{x})}{q_{\phi}(\mathbf{z})}\right]\\
&= \int p_{\theta}(\mathbf{z} \mid \mathbf{x}) \log \frac{p_{\theta}(\mathbf{z} \mid \mathbf{x})}{q_{\phi}(\mathbf{z})}\,d\mathbf{z}\\
&= \int \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{p_{\theta}(\mathbf{x})} \log \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z}) p_{\theta}(\mathbf{x})}\, d\mathbf{z}\\
&= \frac{1}{p_{\theta}(\mathbf{x})}\left(\int p_{\theta}(\mathbf{x}, \mathbf{z}) \left(\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z}) - \log p_{\theta}(\mathbf{x})\right)d\mathbf{z}\right)
\end{align*}
$$
여기까지 전개한 다음 살펴보면, 앞에 붙어 있는 $p_{\theta}(\mathbf{x})$는 최적화 문제와 관련 없는 상수이므로 무시할 수 있다. 마찬가지로, 적분 내부의 첫 번째와 세 번째 항도 $\phi$와 관련 없으므로 무시할 수 있다. 따라서 새로운 목적 함수 $J_{1}'(\phi)$는 다음과 같이 간단해진다.
{{< eqlabel objective-left >}}
$$
J_{1}'(\phi) = -\int p_{\theta}(\mathbf{x}, \mathbf{z}) \log q_{\phi}(\mathbf{z})\,d\mathbf{z}
$$

여기에서 일단 막힌다. 이제 다음 경우를 살펴보자.

### 미지의 분포를 오른쪽에 두기

{{< callout type="Problem" >}}
하나의 관측 데이터 샘플 $\mathbf{x}$가 주어졌을 때 다음 최적화 문제를 풀자.
$$\phi = \argmin_{\phi} D_{\mathrm{KL}} (q_{\phi}(\mathbf{z}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))
$$
{{< /callout >}}

마찬가지로 최적화 문제의 목적 함수를 $J_{2}(\phi)$라 하고 정리해 보자.

{{< eqlabel kl-right >}}
$$
\begin{align*}
J_{2}(\phi) &= D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))\\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})} \left[\log \frac{q_{\phi}(\mathbf{z})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})}\right]\\
&= \int q_{\phi}(\mathbf{z}) \log \frac{q_{\phi}(\mathbf{z})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})}\,d\mathbf{z}\\
&= \int q_{\phi}(\mathbf{z}) \log \frac{q_{\phi}(\mathbf{z}) p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{x}, \mathbf{z})}\, d\mathbf{z}\\
&= \int q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z}) + \log p_{\theta}(\mathbf{x}) - \log p_{\theta}(\mathbf{x}, \mathbf{z})\right)d\mathbf{z}
\end{align*}
$$
여기까지 전개한 다음 살펴보면, 두 번째 항인 $q_{\phi}(\mathbf{z}) \log p_{\theta}(\mathbf{x})$를 $\mathbf{z}$에 대해 적분하면 앞에 있는 $q_{\phi}(\mathbf{z})$가 $1$이 되어 사라진다. 남는 것은 $\log p_{\theta}(\mathbf{x})$인데, 이 항은 최적화 문제에서 무시할 수 있다. 새로운 목적 함수 $J_{2}'(\phi)$는 다음과 같다.
{{< eqlabel objective-right >}}
$$
J_{2}'(\phi) = \int q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z})\right)\,d\mathbf{z}
$$

### 왼쪽 vs 오른쪽
식 {{< eqref objective-left >}}와 식 {{< eqref objective-right >}}를 비교해 보자. 식 {{< eqref objective-right >}}는 피적분식 전체에 $q_{\phi}(\mathbf{z})$가 곱해져 있는데, 이것은 우리가 설계한 확률 분포의 밀도함수이므로 몬테 카를로 근사를 이용해 $J_{2}'(\phi)$를 근사할 수 있다. 반면 식 {{< eqref objective-left >}}의 피적분식에 곱해진 $p_{\theta}(\mathbf{x}, \mathbf{z})$는 $\mathbf{z}$의 확률 분포의 밀도함수가 아니므로 계산하기 난감하다. 그래서 우리의 선택은 오른쪽으로 기울게 된다.

이제 실제로 오른쪽 식이 계산 가능한지 확인해야 한다. 우리가 실제로 필요한 값은 목적 함수의 gradient $\nabla_{\phi} J_{2}'(\phi)$이다.
$$
\begin{align*}
\nabla_{\phi} J_{2}'(\phi) &= \nabla_{\phi} \int q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z})\right)\,d\mathbf{z}\\
&= \int \left(\nabla_{\phi} q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z})\right) + q_{\phi}(\mathbf{z}) \nabla_{\phi}\log q_{\phi}(\mathbf{z}) \right)d\mathbf{z}\\
&= \int \left(\nabla_{\phi} q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z})\right) +  \nabla_{\phi} q_{\phi}(\mathbf{z}) \right)d\mathbf{z}\\
&= \int q_{\phi}(\mathbf{z}) \frac{\nabla_{\phi} q_{\phi}(\mathbf{z})}{ q_{\phi}(\mathbf{z})} \left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z}) \right)d\mathbf{z}\\
&= \mathbb{E}_{\mathbf{z} \sim  q_{\phi}(\mathbf{z})}[\nabla_{\phi} \log q_{\phi}(\mathbf{z})\left(\log q_{\phi}(\mathbf{z})- \log p_{\theta}(\mathbf{x}, \mathbf{z}) \right)]
\end{align*}
$$

위 식의 네 번째 등호에서 log-derivative trick을 이용해 적분을 기댓값으로 바꾸었다. 그리고 네 번째 등호에서 갑자기 맨 마지막에 있던 $\nabla_{\phi} q_{\phi}(\mathbf{z})$가 사라졌는데, 그 이유는 이 식을 적분하면 $0$이 되기 때문이다. 통계학에서 자주 등장하는 항등식으로, 증명은 다음과 같다.
$$
\int \nabla_{\phi} q_{\phi}(\mathbf{z}) = \nabla_{\phi} \int q_{\phi}(\mathbf{z}) = \nabla_{\phi} 1 = 0
$$
이렇게 계산 가능한 식을 얻었으므로, 우리는 마음 놓고 오른쪽 식을 선택할 수 있다.

{{< callout type="Note" >}}
계산 가능한 식을 얻긴 했지만, 이번에도 몬테 카를로 근사의 높은 분산 문제가 우리를 괴롭힌다. $q_{\phi}(\mathbf{z})$에서 $K$개의 샘플 $\mathbf{z}^{(1)}$, $\cdots$, $\mathbf{z}^{(K)}$를 얻은 뒤 몬테 카를로 근사를 적용하면 다음과 같다.
{{< eqlabel high-variance-of-mc >}}
$$
\nabla_{\phi} J_{2}'(\phi) \approx \frac{1}{K} \sum_{k=1}^{K} \nabla_{\phi} \log q_{\phi}(\mathbf{z}^{(k)})\left(\log q_{\phi}(\mathbf{z}^{(k)})- \log p_{\theta}(\mathbf{x}, \mathbf{z}^{(k)}) \right)
$$

이 근사값의 분산이 큰 이유나 해결 방법은 다음 포스트에서 VAE를 다룰 때 자세히 살펴볼 것이다. VAE에서는 **reparametrization trick**으로 이 문제를 해결한다. 이 아이디어는 지금 상황에도 적용할 수 있지만, 일단 넘어가자. 딥러닝 이전의 variational Bayes에서는 비교적 단순한 모델을 다루었기 때문에, 위 식을 몬테 카를로 근사로 계산하기보다는 다른 방법으로 최적화 문제를 풀었다.
{{< /callout >}}

### Mean Field Approximation

Posterior distribution을 근사하는 분포 $q_{\phi}$는 보통 단순한 분포로 설정한다. 실제 posterior인 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$에 대한 표현력은 부족할지 몰라도, 계산은 매우 편리해진다. $p_{\theta}$도 단순한 모델일 경우 목적 함수를 근사 없이 해석적으로 구하는 것도 가능해진다.

$q_{\phi}(\mathbf{z})$에 대해 많이 하는 가정 중 하나는, 잠재 변수 $\mathbf{z}$를 $(z_{1}, \cdots, z_{M})$으로 나타낼 수 있을 때 이들의 posterior가 모두 독립이라는 것이다. 이를 **mean field approximation**이라고 하며, 식으로 나타내면 다음과 같다.
$$
q_{\phi}(\mathbf{z}) = \prod_{m=1}^{M} q_{\phi_{m}} (z_{m})
$$

딥러닝 이전에는 비교적 단순한 모델들을 다루었기 때문에, 이와 같은 가정을 도입해 해석적인 해를 구할 수 있는 경우가 많다. 심지어는 매개변수 $\phi$ 없이 모든 확률 분포 중에서 최적해를 찾는 것도 가능하다. 딥러닝에서는 복잡한 신경망으로 확률 분포를 모델링하기 때문에 mean field approximation은 그리 효과적이지 않으므로, 더 이상 언급하지 않고 넘어가겠다.

### Evidence Lower Bound (ELBO)

[첫 포스트](../01-overview/#divergence-minimization)에서 KL divergence 최소화가 MLE와 동일하다는 것을 보였다. 여기에서도 그럴까? 그때 우리가 보인 것은 미지의 분포가 KL divergence의 왼쪽에 있는 상황이었는데, 지금은 오른쪽에 있으므로 그때의 결과를 활용할 수는 없다. 따라서 $q_{\phi}$를 얻는 과정을 MLE로 볼 수 없다.

한편, 여기에서는 $\mathbf{x}$의 marginal likelihood인 $p_{\theta}(\mathbf{x})$와 KL divergence 사이에 흥미로운 관계가 성립한다. 먼저 식 {{< eqref kl-right >}}의 결과를 가져와 다음과 같이 다시 써 보자.
$$
\begin{align*}
D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))
&= \int q_{\phi}(\mathbf{z}) \left(\log q_{\phi}(\mathbf{z}) + \log p_{\theta}(\mathbf{x}) - \log p_{\theta}(\mathbf{x}, \mathbf{z})\right)d\mathbf{z}\\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})} [\log q_{\phi}(\mathbf{z}) + \log p_{\theta}(\mathbf{x}) - \log p_{\theta}(\mathbf{x}, \mathbf{z})]\\
&=  \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})} [\log q_{\phi}(\mathbf{z}) - \log p_{\theta}(\mathbf{x}, \mathbf{z})] + \log p_{\theta}(\mathbf{x})
\end{align*}
$$

$\log p_{\theta}(\mathbf{x})$가 자연스럽게 튀어나왔다! 그리고 $\log p_{\theta}(\mathbf{x})$를 제외한 나머지 부분은 식 {{< eqref objective-right >}}에서 본 목적 함수이다. 이제 샘플 $\mathbf{x}$를 고정한 상황이고 $\mathbf{z}$의 posterior가 매개변수 $\theta$를 사용하는 모델에서 나온 것임을 강조해서, 목적 함수를 $\mathcal{L}(\phi; \mathbf{x}, \theta)$라고 다시 쓰자. 그리고 편의상 목적 함수의 부호를 바꿔, $\mathcal{L}(\phi; \mathbf{x}, \theta) = -J_{2}'(\phi)$로 정의하자. 그러면 다음 관계가 성립한다는 것을 바로 알 수 있다.
{{< eqlabel elbo >}}
$$
\log p_{\theta}(\mathbf{x}) = D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})) + \mathcal{L}(\phi; \mathbf{x}, \theta)
$$

목적 함수의 부호가 바뀌었으므로, 최적화 문제는 $\mathcal{L}(\phi; \mathbf{x}, \theta)$를 최대화하는 문제가 된다. 식 {{< eqref elbo >}}에서 $\log p_{\theta}(\mathbf{x})$는 고정된 값이므로, 목적 함수 $\mathcal{L}$과 KL divergence의 합은 일정하다. 따라서 목적 함수의 최대화가 KL divergence를 최소화한다는 결과를 다시 한번 확인할 수 있다. 우리는 애초에 KL divergence 최소화가 목표였으므로, 여기까지는 당연하다.

그런데 식 {{< eqref elbo >}}는 추가적인 결과를 하나 더 제공한다. 바로 KL divergence가 $0$에 가까워질수록 목적 함수 $\mathcal{L}$은 $\log p_{\theta}(\mathbf{x})$에 가까워진다는 것이다. 그리고 KL divergence는 항상 $0$ 이상이기 때문에, 목적 함수가 아무리 커져도 $\log p_{\theta}(\mathbf{x})$보다 커질 수는 없다. 즉, 다음 관계가 성립한다.
$$
\log p_{\theta}(\mathbf{x}) \ge \mathcal{L}(\phi; \mathbf{x}, \theta)
$$

이는 **목적 함수를 최대화하는 것이 marginal likelihood를 근사하는 것과 같다**는 의미이다. 또한, 위 부등식에 의해 목적 함수 $\mathcal{L}$은 marginal likelihood의 **lower bound**가 된다. 베이지안 추론에서는 marginal likelihood를 evidence라는 이름으로도 부르므로, 이렇게 정의된 함수 $\mathcal{L}$을 **evidence lower bound (ELBO)** 라고 부른다.

ELBO $\mathcal{L}$이 marginal likelihood $\log p_{\theta}(\mathbf{x})$를 lower bound로써 근사한다는 사실은 VAE에서 결정적인 역할을 한다. 다음 포스트에서 살펴보겠지만, ELBO는 VAE에서도 목적 함수로 활용된다.

### Limitations of Variational Bayes

Variational Bayes는 MCMC와 달리 반복적인 샘플링 없이 최적화 문제를 풀어 posterior를 근사한다는 장점이 있다. 하지만 딥러닝에 적용하기에는 여전히 한계가 있다.

가장 큰 문제는 각 샘플 $\mathbf{x}$마다 매개변수 $\phi$를 따로 최적화해야 한다. 이 문제는 MCMC에서도 동일하게 발생했었다. 데이터가 $N$개이면 $N$개의 서로 다른 최적화 문제를 풀어야 하므로, 샘플의 수가 많아지면 계산 비용이 급격히 증가한다. 또한, 학습 과정에서 $\theta$가 업데이트될 때마다 모든 $\phi$를 다시 최적화해야 한다. 이것이 variational Bayes를 바로 딥러닝에 적용할 수 없는 결정적인 이유이다.

다른 문제로, $q_{\phi}(\mathbf{z})$를 단순하게 설정하면 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 잘 표현하지 못하고, 표현력이 높은 복잡한 분포를 사용하면 최적화가 어려워진다는 딜레마가 있다. 이 점은 딥러닝에서 $q_{\phi}$를 신경망으로 모델링하면 어느 정도 해결할 수 있다. 대신 이렇게 정의된 $q_{\phi}$를 최적화하려면 경사 하강법이 필요하고, 그래서 몬테 카를로 근사를 통해 gradient를 계산해야 하는데, 그러면 식 
{{< eqref high-variance-of-mc >}}에서 보았던 높은 분산 문제를 겪게 된다. 딥러닝에서 variational bayes의 아이디어를 적용하려면 이 문제를 반드시 해결해야 한다.

# 정리

이번 포스트에서는 latent variable model을 구체적으로 살펴보고, marginal likelihood인 $p_{\theta}(\mathbf{x})$를 계산하기 어렵다는 문제를 확인했다. 이를 해결하기 위해서는 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 구해야 하는데, 여기에 대한 세 가지 방법을 살펴보았다.

- **E-M algorithm**: Posterior를 해석적으로 구할 수 있는 단순한 모델에서 강력하지만, 신경망 기반의 복잡한 모델에는 적용하기 어렵다.
- **MCMC**: Posterior의 정규화 상수를 몰라도 샘플링이 가능하지만, 수렴이 느리고 매 step마다 긴 Markov chain을 돌려야 하므로 학습에 활용하기 어렵다.
- **Variational Bayes**: Posterior를 근사하는 분포 $q_{\phi}(\mathbf{z})$를 최적화하는 방법으로, ELBO라는 목적 함수를 활용한다. 여전히 복잡한 모델에 적용하기 어렵다.

다음 포스트에서는 딥러닝에서 이러한 한계들을 극복하는 방법에 대해 알아볼 것이다.



{{< reflist >}}
{{< refitem 1 >}}
Kingma, Diederik P., and Welling, Max. "[Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114)". *arXiv preprint*, 2013.
{{< /refitem >}}
{{< refitem 2 >}}
Kingma, Diederik P., and Ba, Jimmy. "[Adam: A method for stochastic optimization](https://arxiv.org/abs/1412.6980)". *arXiv preprint*, 2014.
{{< /refitem >}}
{{< refitem 3 >}}
Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm". *Journal of the royal statistical society: series B (methodological)*, 39.1: 1-22, 1977.
{{< /refitem >}}
* 이 논문을 직접 참고하지는 않았다.
{{< refitem 4 >}}
Wikipedia, "[Expectation-maximization algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)".
{{< /refitem >}}
{{< refitem 5 >}}
Wikipedia, "[Rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling)".
{{< /refitem >}}
{{< refitem 6 >}}
Wikipedia, "[Metropolis-Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm)".
{{< /refitem >}}
{{< refitem 7 >}}
Wikipedia, "[Variational Bayesian methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)".
{{< /refitem >}}
{{< /reflist >}}
