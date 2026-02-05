---
title: "Generative Models - 01. Overview"
date: 2026-02-03
tags: ["machine-learning", "generative-models"]
math: true
---

## 생성 모델로 풀고 싶은 문제

먼저 생성 모델(Generative Model)로 풀고 싶은 문제를 생각해 보자.

- Text Generation
  - 텍스트로 질문이 주어졌을 때, 답변을 텍스트로 생성
  - 문맥에 맞는 문장 완성, 요약, 번역
  - 코드 생성 및 자동 완성
- Image Generation
  - 실제 사진과 구분이 안 되는 이미지 생성
  - 이미지에 대한 설명이 텍스트로 주어졌을 때 적절한 이미지 생성
  - 주어진 이미지를 다른 스타일로 변환
- Audio Generation
  - 텍스트를 자연스러운 음성으로 변환
- Video Generation
  - 텍스트나 이미지로부터 영상 생성
- 3D Generation
  - 텍스트나 이미지로부터 3D 모델 생성

이처럼 다양한 문제들이 있지만, 결국 '어떤 조건이 주어졌을 때 조건을 만족하는 데이터를 생성한다'라는 공통점을 가지고 있다. 또 다른 공통점으로, 우리가 생성해야 할 데이터의 예시(샘플)를 충분히 가지고 있지만, 그 예시들이 어떤 메커니즘으로 생겨났는지는 알 수 없다는 점이 있다. 예를 들어, 실제 사진과 구분이 안 되는 이미지를 생성하는 문제의 경우, 우리는 실제 사진을 충분히 많이 수집할 수 있지만 그 사진을 구성하는 픽셀들의 값이 왜 그렇게 정해졌는지, 다른 픽셀들과의 관계가 어떻게 되는지는 설명할 수 없다. 또한, 이미지에 대한 설명이 텍스트로 주어졌을 때 이미지를 생성하는 문제를 생각해 보면, 우리는 충분히 많은 이미지를 수집할 수 있고, 수집한 이미지에 대한 설명을 텍스트로 작성할 수 있다. 하지만 이미지에 대한 설명을 보고 이미지를 재구성하는 방법을 명시적으로 설명하는 것은 불가능에 가깝다. 우리는 이런 문제를 충분한 수의 데이터와 적절한 모델링만으로 해결하려는 것이다.

## 문제의 모델링

문제를 수학적으로 모델링해 보자. 먼저 우리가 생성해야 할 데이터를 모델링하자. 생성할 데이터의 가능한 모든 값의 범위, 즉 도메인을 생각하고, 도메인의 원소를 $\mathbf{x}$로 나타내자. 만약 생성할 데이터에 조건이 붙어 있는 경우, 조건을 $\mathbf{c}$로 나타내자.

다음으로 우리에게 주어진 것을 모델링하자. 우리는 데이터 $\mathbf{x}$ 또는 데이터와 조건의 조합 $(\mathbf{x}, \mathbf{c})$의 샘플을 충분히 확보할 수 있다. 샘플을 확보하는 방법은 매우 다양할 수 있지만, 이것을 수학적으로 모델링할 때는 우리가 원할 때 $\mathbf{x}$ 또는 $(\mathbf{x}, \mathbf{c})$를 던져 주는 oracle이 있다고 가정하는 것이 편리하다. 또한, 이 oracle은 어떤 **확률 분포**에서 샘플을 랜덤하게 골라서 우리에게 던져 준다고 가정하는 것이 편리하다. 이 확률 분포를 실제 데이터의 확률 분포라는 의미에서 $p_{\mathrm{data}}$라고 부르자. 일반적으로 $N$ 개의 샘플 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$ (또는 조건을 포함한 샘플 $(\mathbf{x}_{1}, \mathbf{c}_{1})$, $\cdots$, $(\mathbf{x}_{N}, \mathbf{c}_{N})$)이 있을 때, 이 샘플들은 모두 동일한 확률 분포 $p_\mathrm{data}$에서 독립적으로 랜덤 샘플링되었다고 가정한다 (IID: independent and identically distributed). 마찬가지로 편리해서 그렇다.

다음으로 우리의 목표를 모델링하자. 먼저 조건 $\mathbf{c}$가 주어지지 않은 경우, 즉 실제 데이터와 구분되지 않는 데이터 $\mathbf{x}$를 생성하고자 하는 문제를 보자. 이때 우리의 목표는 oracle을 거치지 않고 oracle이 가진 확률 분포 $p_{\mathrm{data}}(\mathbf{x})$에서 데이터 $\mathbf{x}$를 샘플링하는 것이 된다. 하지만 우리가 oracle에서 얻은 $N$ 개의 샘플만으로 $p_{\mathrm{data}}(\mathbf{x})$를 정확히 알아내는 것은 물론 불가능하다. 따라서 우리는 주어진 샘플을 이용해 $p_{\mathrm{data}}(\mathbf{x})$와 **최대한 가까운 확률 분포** $p(\mathbf{x})$를 얻은 다음, $p$에서 $\mathbf{x}$를 샘플링하는 것을 목표로 하게 된다. 즉,
$$p(\mathbf{x}) \approx p_{\mathrm{data}}(\mathbf{x})$$
를 만족하는 $p(\mathbf{x})$를 찾고, $\mathbf{x} \sim p(\mathbf{x})$를 수행하는 문제가 된다.

조건 $\mathbf{c}$가 주어지는 경우에는 조건부 확률을 도입해 마찬가지로 모델링할 수 있다. 이 경우
$$p(\mathbf{x} \mid \mathbf{c}) \approx p_{\mathrm{data}}(\mathbf{x} \mid \mathbf{c})$$
를 만족하는 $p(\mathbf{x} \mid \mathbf{c})$를 찾고, $\mathbf{c}$가 주어졌을 때 $\mathbf{x} \sim p(\mathbf{x} \mid \mathbf{c})$를 수행하는 문제가 된다. 조건 $\mathbf{c}$가 주어진 문제는 conditional generation이라고 하는데, 물론 조건이 없는 일반적인 generation보다 어려운 문제이다. 당분간은 일반적인 generation에 초점을 맞추고, conditional generation은 나중에 살펴보자.

## 최대한 가까운 확률 분포란 무엇을 의미하는가?

이제 방금 사용한 '최대한 가까운 확률 분포'라는 표현을 수학적으로 정의할 필요가 있다. 여기에서 우리의 목표를 다시 적어 보면,

{{< callout type="Goal" >}}
$p_{\mathrm{data}}(\mathbf{x})$에서 IID 샘플링한 데이터 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$이 있다. 이것을 이용해 $p(\mathbf{x}) \approx p_\mathrm{data} (\mathbf{x})$인 $p$를 찾자.
{{< /callout >}}

이 문제의 어려움은 실제 확률 분포의 밀도함수의 값 $p_{\mathrm{data}}(\mathbf{x})$을 알아낼 수 없다는 점에 있다. 실제 분포와 관련해서 우리가 접근할 수 있는 것은 오직 $N$개의 샘플 뿐이다. 하지만, 우리가 궁극적으로 원하는 것은 우리 마음대로 샘플링이 가능한 분포 $p$이고, 샘플링을 하기 위해서는 밀도함수 $p(\mathbf{x})$에 대한 정보를 어떻게든 가지고 있어야 한다. (밀도함수 $p(\mathbf{x})$의 정보를 어디까지 알아내야 샘플링이 가능할지에 대해서는 아래에서 다시 살펴보자.)

{{< callout type="Note" >}}
사실 $N$이 매우 크면 샘플 $\mathbf{x}_{i}$를 통해 밀도함수 $p_{\mathrm{data}}$를 충분히 근사할 수 있는 것은 맞다. 하지만 우리가 관심 있는 이미지나 텍스트처럼 고차원의 데이터가 주어졌을 경우 우리에게 유용한 수준까지 밀도함수를 근사하기 위해서는 어마어마하게 큰 $N$이 필요하다. 따라서 '밀도함수에 대한 정보는 얻을 수 없다'라고 보아야 한다.
{{< /callout >}}

### 매개화된 확률 분포

실제 분포에 대한 정보가 워낙 제한되어 있기 때문에, 우리가 해볼 수 있는 가정은 몇 가지 없다. 그 중 하나는 '우리가 가지고 있는 샘플 $\mathbf{x}_{i}$에서는 밀도함수 $p_{\mathrm{data}}(\mathbf{x}_{i})$가 높지 않을까?'라는 것이다. 이 가정은 너무나 말이 되는 것이, 밀도함수 $p_{\mathrm{data}}(\mathbf{x})$가 높을수록 $\mathbf{x}$가 샘플로 나올 가능성이 높아지기 때문이다. 그런 의미에서 우리는 샘플에서 $p$의 밀도함수의 값, 즉 $p(\mathbf{x}_{1})$, $\cdots$, $p(\mathbf{x}_{N})$을 최대화하면 $p$가 $p_{\mathrm{data}}$에 가까워지지 않을까? 라고 기대할 수 있다.

그런데 여기에서 극단적인 예시를 하나 만들어 보자. $p$가 다음과 같이 정의된 이산 확률 분포라고 하자.
$$p(\mathbf{x}_{i}) = \frac{1}{N} \; , \qquad i = 1, \cdots , N$$
즉, $p$는 우리가 가지고 있는 $N$개의 샘플 중 하나를 균일한 확률로 샘플링할 수 있고, 다른 데이터는 샘플링할 수 없는 확률 분포이다. 이러한 확률 분포 $p$는 명백히 $p(\mathbf{x}_{1})$, $\cdots$, $p(\mathbf{x}_{N})$을 최대화하려는 목표와 가깝다. 하지만 이런 방식으로는 이미 가지고 있는 샘플 외의 새로운 데이터는 얻을 수 없기에, 우리의 실제 목표와는 매우 멀다. 그리고 ($N$이 어마어마하게 크지 않은 이상) '$p$가 $p_{\mathrm{data}}$에 유사하다'라는 주장에 동의하는 사람도 없을 것이다.

위의 예시를 통해, 아무런 제약 없이 $p$를 찾으면 원하지 않는 결과가 나올 수 있다는 것을 알 수 있다. 따라서 $p$가 가질 수 있는 형태를 미리 제한할 필요가 있다. 이를 위해 매개변수(parameter) $\phi$를 도입하여, $\phi$의 값에 따라 밀도함수의 형태가 결정되도록 하자. 이렇게 매개화된 확률 분포를 $p_{\phi}$로 표기한다. 이제 우리의 목표는 $p_{\phi}$가 $p_{\mathrm{data}}$에 최대한 가까워지도록 하는 $\phi$를 찾는 것이 된다. $p$의 형태에 제한을 두었기 때문에, 이렇게 찾은 $p_{\phi}$는 $p_{\mathrm{data}}$의 형태를 충분히 표현하지 못할 수도 있다. 하지만 우리가 허용한 형태 중 실제 데이터를 가장 잘 설명하는 분포를 찾은 것이기 때문에, 충분히 유용하다.

{{< callout type="Note" >}}
매개화된 확률 분포의 대표적인 예시는 정규분포이다. 다음은 $\phi = (\mu, \Sigma)$를 매개변수로 갖는 $d$차원 정규분포의 밀도함수이다.

$$p_{\phi}(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^{\top} \Sigma^{-1} (\mathbf{x} - \mu) \right)$$
{{< /callout >}}

### Maximum Likelihood Estimation

이제 '최대한 가까운'을 정의할 차례이다. 앞에서의 논의는 여전히 유효한데, 샘플에서의 밀도함수 $p_{\phi}(\mathbf{x}_{i})$가 클수록 $p_{\mathrm{data}}$에 가까운 확률분포라는 것이다. $p_{\phi}(\mathbf{x}_{i})$라는 식은 원래 매개변수 $\phi$를 가진 확률 분포에서 $\mathbf{x}_{i}$의 밀도라는 의미이지만, 여기에서는 반대로 샘플 $\mathbf{x}_{i}$가 주어졌을 때 매개변수 $\phi$가 얼마나 '좋은' 매개변수인지 평가하는 역할을 하고 있다. 이것을 샘플 $\mathbf{x}_{i}$가 주어졌을 때 매개변수 $\phi$의 가능도(likelihood)라고 하고, $\mathcal{L}(\phi \mid \mathbf{x}_{i})$로 나타낸다.
$$\mathcal{L}(\phi \mid \mathbf{x}_{i}) = p_{\phi}(\mathbf{x}_{i})$$

$N$개의 샘플이 $p_{\mathrm{data}}$로부터 얻은 IID 샘플이므로, $N$개의 샘플 전체가 주어졌을 때 $\phi$의 가능도를 다음과 같이 정의할 수 있다. 독립성에 의해 각각의 샘플에 의한 가능도의 곱으로 나타난다.
$$\mathcal{L}(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) = \prod_{i} p_{\phi} (\mathbf{x}_{i}) = \prod_{i} \mathcal{L}(\phi \mid \mathbf{x}_{i})$$

드디어 우리의 목표를 제대로 정의할 수 있다. 가능도가 클수록 좋은 매개변수이므로, 우리는 다음을 만족하는 $\phi$를 찾고 싶다.
$$\begin{aligned}
\phi &= \argmax_{\phi} \; \mathcal{L}(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) \\
&= \argmax_{\phi} \; \prod_{i} p_{\phi}(\mathbf{x}_{i})
\end{aligned}$$

위 식의 우변에는 $p_{\mathrm{data}}$의 밀도함수와 관련된 항이 등장하지 않는다. 오직 샘플 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$과 우리가 이미 형태를 알고 있는 함수 $p_{\phi}$ 뿐이다. 따라서 우리가 해결할 수 있는 최적화 형태의 문제를 얻었다.

곱보다는 합이 다루기 편리하므로, 로그가능도(log-likelihood)를 $\ell(\phi \mid \mathbf{x}) = \log \mathcal{L}(\phi \mid \mathbf{x})$로 정의하면 위 식을 더 편하게 다룰 수 있다.

$$\begin{aligned}
\phi &= \argmax_{\phi} \; \ell(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N})\\
&= \argmax_{\phi} \; \log \left( \prod _{i} p_{\phi}(\mathbf{x}_{i}) \right)\\
&= \argmax_{\phi} \; \sum _{i} \log p_{\phi}(\mathbf{x}_{i})
\end{aligned}$$

확률 분포의 매개변수를 구하기 위해 위와 같은 최적화 문제를 활용하는 방법을 최대 가능도 추정(MLE, Maximum Likelihood Estimation)이라 한다. 20세기 초 R. Fisher가 정립한 이후 기계 학습에도 활용되고 있다.

{{< callout type="Note" >}}
MLE에서는 모든 매개변수를 평등하게 대하고 있다. 하지만 베이지안 통계에서는 매개변수 $\phi$를 확률 변수로 간주한 뒤, $\phi$의 사전 확률 분포(prior distribution)을 고려하기도 한다. 이 방법을 maximum a posteriori estimation (MAP)라 한다.

MLE에서 어떤 매개변수 $\phi$가 얼마나 '좋은지' 나타내는 척도로 가능도 $\mathcal{L}(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) = p_{\phi}(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N})$을 사용했다. MAP에서는 이와 같은 척도로 조건부 확률 $p(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N})$를 활용한다. 이것이 가능한 이유는 매개변수 $\phi$를 확률 변수로 보고 있기 때문이다.

$\phi$의 사전 확률 분포를 $p(\phi)$라 하면, 이 조건부 확률은 베이즈 정리에 의해 다음과 같이 구할 수 있다.
$$
p(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) = \frac{p(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \mid \phi) \;p(\phi)}{\int p(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \mid \phi) \;p(\phi) \; d\phi}
$$

$p(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \mid \phi)$는 다름 아닌 $p_{\phi}(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N})$이다. 분모의 적분은 너무 복잡해서 계산할 수 없지만, $\phi$와 무관한 상수이므로 최적화 문제에서 무시할 수 있다. 이제 이 값을 최대화하는 $\phi$를 구하는 최적화 문제를 생각해 보면,

$$\begin{aligned}
\phi &= \argmax_{\phi} \; p(\phi \mid \mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) \\
&= \argmax_{\phi} \frac{p(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \mid \phi) \;p(\phi)}{\int p(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N} \mid \phi) \;p(\phi) \; d\phi} \\
&= \argmax_{\phi} \; p_{\phi}(\mathbf{x}_{1}, \cdots, \mathbf{x}_{N}) \; p(\phi)\\
&= \argmax_{\phi} \; p(\phi) \prod_{i} p_{\phi}(\mathbf{x}_{i})\\
&= \argmax_{\phi} \left(\log p(\phi) + \sum_{i} \log p_{\phi}(\mathbf{x}_{i}) \right)
\end{aligned}
$$

마지막 등호에서는 MLE에서와 마찬가지로 로그를 씌웠다. 이 식과 MLE에서 최적화하는 식을 보면, $\log p(\phi)$라는 항이 더해졌음을 알 수 있다. 이 항은 $\phi$의 확률 밀도가 높은 곳에서는 $\phi$에 advantage를 주고, 낮은 곳에서는 penalty를 주는 역할을 하고 있다.

이 항을 조금 더 살펴보자. $\phi$의 사전 확률 분포가 정규분포라고 가정하자. $d$ 개의 원소로 이루어진 매개 변수 $\phi$가 정규 분포를 따르고 각 원소의 분산이 $\sigma^{2}$이라 가정하면 (즉, $\phi \sim \mathcal{N}(0, \sigma^{2} I)$라면), 밀도함수 $p(\phi)$는 다음과 같다.

$$p(\phi) = \frac{1}{(2\pi\sigma^{2})^{d/2}} \exp\left( -\frac{\|\phi\|^2}{2\sigma^{2}} \right)$$

$\log p(\phi)$는 다음과 같다.

$$\log p(\phi) = - \frac{\|\phi\|^2}{2\sigma^{2}} - \frac{d}{2} \log(2\pi\sigma^{2})$$

두 번째 항은 $\phi$와 관련 없는 상수이므로 최적화 문제에서 중요하지 않다. 첫 번째 항은 $\|\phi\|^{2}$에 비례하는 항인데, 다름이 아니라 우리가 L2 regularization이라 부르는 항과 동일하다.

따라서 MLE에 L2 regularization을 적용하면 매개변수가 정규 분포를 따르는 MAP와 동일해진다는 것을 알 수 있다. 일반적으로 딥러닝에서 볼 수 있는 매개변수의 정규화(regularization)는 매개변수의 사전 확률 분포를 고려한 것이다.

{{< /callout >}}

### Divergence Minimization

지금까지 살펴본 MLE는 실제 확률 분포 $p_{\mathrm{data}}$와 가까운 $p_{\phi}$를 찾는 여러 가지 방법 중 하나이다. 다른 방법도 살펴보자. 기계 학습에서는 두 확률 분포 $p$와 $q$가 얼마나 다른지, 즉 괴리도(divergence)를 측정하는 함수 $D(p \| q)$를 정의하고, $D(p_{\mathrm{data}} \| p_{\phi})$를 최소화하는 방법을 주로 사용한다. 이러한 함수 중 가장 많이 쓰이는 것은 Kullback-Leibler divergence (KL divergence)로, 다음과 같이 정의된다.
$$D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} \left[\log\frac{p(\mathbf{x})}{q(\mathbf{x})}\right]$$

KL divergence는 두 확률 분포 $p$와 $q$가 같을 때 (정확히는 거의 어디서나(almost everywhere) $p = q$일 때) 0이고, 두 분포의 차이가 커질수록 값이 증가하는 성질이 있다. 우리는 $p_{\phi}$를 $p_{\mathrm{data}}$에 가깝게 만들고 싶으므로, 다음과 같은 최적화 문제를 풀어야 한다.

$$\begin{aligned}
\phi &= \argmin_{\phi} \; D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\phi}) \\
&= \argmin_{\phi} \; \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[\log\frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\phi}(\mathbf{x})}\right]
\end{aligned}
$$

앞에서 실제 데이터의 밀도함수 $p_{\mathrm{data}}(\mathbf{x})$를 알 수 없다고 했는데, 이 식을 어떻게 풀 수 있을까? 먼저 $\phi$에 관련 없는 항을 분리해야 한다. 아래 식에서 최소화 문제의 부호를 뒤집어 최대화 문제로 바꾼 것에 유의해야 한다.

{{< eqlabel kl-objective >}}
$$\begin{aligned}
\phi &= \argmin_{\phi} \; \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[\log p_{\mathrm{data}}(\mathbf{x}) - \log p_{\phi}(\mathbf{x})\right] \\
&= \argmax_{\phi} \; \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[\log p_{\phi}(\mathbf{x})\right]
\end{aligned}$$

이제 우리가 가지고 있는 샘플 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$을 이용해 기댓값을 근사하면 다음과 같이 풀 수 있는 식을 얻는다.

{{< eqlabel mle-approx >}}
$$\phi = \argmax_{\phi} \; \frac{1}{N} \sum_{i=1}^{N} \log p_{\phi}(\mathbf{x}_{i})$$

이것은 앞에서 살펴본 MLE와 동일한 식이다. 즉, KL divergence를 최소화하는 것과 MLE는 같은 최적화 문제를 풀고 있다. 통계학이나 딥러닝에서는 divergence를 최소화하는 것을 더 자연스러운 접근으로 본다. 따라서 위의 논의는 MLE에 정당성을 부여해 주는 것이 된다.

KL divergence 외에도 다양한 선택지가 있다. 예를 들어, $f: \mathbb{R}^{+} \rightarrow \mathbb{R}$가 $f(1) = 0$을 만족하는 볼록함수일 때 다음과 같이 정의된 $f$-divergence를 사용할 수 있다.

$$D_{f}(p \| q) = \mathbb{E}_{\mathbf{x} \sim q(\mathbf{x})} \left[ f\left( \frac{p(\mathbf{x})}{q(\mathbf{x})} \right) \right]$$

KL divergence는 $f(t) = t \log t$인 $f$-divergence이다. 증명은 다음과 같다.

$$\begin{aligned}
D_{f}(p \| q) &= \mathbb{E}_{\mathbf{x} \sim q(\mathbf{x})} \left[ f\left( \frac{p(\mathbf{x})}{q(\mathbf{x})} \right) \right] \\
&= \mathbb{E}_{\mathbf{x} \sim q(\mathbf{x})} \left[ \frac{p(\mathbf{x})}{q(\mathbf{x})} \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \right] \\
&= \int q(\mathbf{x}) \cdot \frac{p(\mathbf{x})}{q(\mathbf{x})} \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} \\
&= \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} \\
&= \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} \left[ \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \right] \\
&= D_{\mathrm{KL}}(p \| q)
\end{aligned}$$

## 최적화 문제를 어떻게 푸는가?

앞 절에서 우리는 '실제 분포와 가까운 확률 분포를 찾는 문제'를 매개변수 $\phi$에 대한 최적화 문제로 바꾸었다. 이러한 최적화 문제를 해결하는 과정을 **학습**이라고 한다. 

MLE나 KL divergence를 최소화하는 문제 등을 살펴보았는데, 모두 다음과 같은 형태로 쓸 수 있다.

$$
\phi = \argmin_{\phi} \; J(\phi)
$$

여기에서 $J(\phi)$는 목적 함수(objective function)이다. 예를 들어, MLE의 예시에서는 $J(\phi) = -\sum_{i} \log p_{\phi}(\mathbf{x}_{i})$이다.

최적화 문제를 해결하는 방법은 다양하다. 예를 들어 라그랑주 승수법, 담금질 기법(simulated annealing) 등이 있다. 딥러닝에서는 경사 하강법(gradient descent method)과 그 변형 알고리즘들을 사용하는 것이 가장 적합하다.

### Gradient Descent

경사 하강법은 현재 매개변수 $\phi$에서 목적 함수의 기울기인 $\nabla_{\phi} J(\phi)$를 계산한 뒤, 기울기의 반대 방향으로 $\phi$를 조금씩 업데이트하는 방법이다.

{{< eqlabel gradient-descent >}}
$$\phi \leftarrow \phi - \eta \nabla_{\phi} J(\phi)$$

여기서 $\eta > 0$는 한 번에 얼마나 이동할지를 결정하는 하이퍼파라미터이다. 이 업데이트를 반복하면 $J(\phi)$가 극솟값에 수렴하게 된다.

경사 하강법에 대해 구체적으로 서술하지는 않을 것이다. 하지만 여기에서 언급한 이유는 최적화 문제를 풀기 위해서는 $J$가 $\phi$에 대해 미분 가능해야 한다는 점을 강조하기 위해서이다. $J$에는 보통 $\log p_{\phi}(\mathbf{x}_{i})$가 포함되어 있으므로, $p_{\phi}(\mathbf{x}_{i})$가 $\phi$로 미분가능해야 한다. 이것은 우리가 설정한 매개변수화된 확률 분포 $p_{\phi}$가 만족해야 하는 중요한 조건이다.

### Monte Carlo Approximation

최적화 문제를 푸는 방법과 관련해서 언급할 내용이 한 가지 더 있다. 딥러닝에서의 목적 함수는 식 {{< eqref kl-objective >}}와 같이 기댓값을 포함하고 있는 경우가 많다. 일반적으로 $\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} [f(\mathbf{x})]$와 같은 식의 값을 정확히 계산하는 것은 $p$와 $f$가 단순하지 않은 이상 불가능하다. 왜냐하면 다음과 같은 복잡한 적분 형태로 표현되기 때문이다.
{{< eqlabel expectation-integral >}}
$$\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} [f(\mathbf{x})] = \int f(\mathbf{x}) p(\mathbf{x})\; d\mathbf{x}$$
하지만 $p(\mathbf{x})$에서 샘플 $\mathbf{x}_{1}$, $\cdots$, $\mathbf{x}_{N}$을 추출할 수 있다면, 기댓값을 다음과 같이 근사할 수 있다.

$$\mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} [f(\mathbf{x})] \approx \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}_{i}), \quad \mathbf{x}_{i} \sim p(\mathbf{x})$$

이러한 방법을 몬테 카를로 근사(Monte Carlo approximation)라 한다. 큰 수의 법칙(LLN, Law of Large Numbers)에 의해 $N \to \infty$일 때 우변이 좌변에 수렴하는 것이 보장된다. 앞에서 식 {{< eqref mle-approx >}}을 식 {{< eqref kl-objective >}}의 기댓값을 $N$개의 샘플로 근사하여 얻은 것도 같은 원리이다.

사실 우리는 식 {{< eqref gradient-descent >}}을 적용한 경사 하강법으로 최적화 문제를 풀어야 하므로, 식 {{< eqref kl-objective >}}의 기댓값 자체가 아니라 그것을 $\phi$로 미분한 결과가 필요하다. 다행히, 이 값도 마찬가지로 몬테 카를로 근사를 통해 구할 수 있다.

$$\nabla_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\log p_{\phi}(\mathbf{x})] = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\nabla_{\phi} \log p_{\phi}(\mathbf{x})] \approx \frac{1}{N} \sum_{i=1}^{N} \nabla_{\phi} \log p_{\phi}(\mathbf{x}_{i})$$

몬테 카를로 근사를 사용하기 위해서는 (1) 기댓값의 확률 분포에서 샘플링이 가능하고, (2) 피적분함수 $f(\mathbf{x})$의 값을 계산할 수 있어야 한다. 이 조건 중 하나라도 만족하지 못하면 몬테 카를로 근사를 적용할 수 없다.

예를 들어, 앞에서 우리는 $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\phi})$를 최소화하는 문제를 살펴보았다. 반대 방향의 KL divergence $D_{\mathrm{KL}}(p_{\phi} \| p_{\mathrm{data}})$를 최소화하는 문제를 생각해 보자.

$$D_{\mathrm{KL}}(p_{\phi} \| p_{\mathrm{data}}) = \mathbb{E}_{\mathbf{x} \sim p_{\phi}(\mathbf{x})} \left[\log \frac{p_{\phi}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x})}\right]$$

이 식의 기댓값은 $p_{\phi}$에 대한 것이므로 $p_{\phi}$에서 샘플링이 가능하다면 첫 번째 조건은 만족할 수 있다. 하지만 피적분함수에 $p_{\mathrm{data}}(\mathbf{x})$가 포함되어 있고, 우리는 $p_{\mathrm{data}}$의 밀도함수 값을 알 수 없으므로 두 번째 조건을 만족하지 못한다. 따라서 이 식에는 몬테 카를로 근사를 적용할 수 없다.

정리하면, 최적화 문제를 풀기 위해서는 목적 함수 $J(\phi)$가 $\phi$에 대해 미분해 $\nabla_{\phi} J(\phi)$를 구할 수 있어야 한다. 만약 $\nabla_{\phi} J(\phi)$에 기댓값이 있다면 몬테 카를로 근사를 적용할 수 있어야 한다.

## 확률 분포 $p_{\phi}$를 어떻게 모델링해야 하는가?

아직 $p_{\phi}$가 어떤 형태의 분포인지에 대해서는 구체적으로 이야기하지 않았다. $p_{\phi}$를 설계할 때 고려해야 할 점은 다음과 같다.

1. **표현력**: $p_{\phi}$는 복잡한 실제 분포 $p_{\mathrm{data}}$를 충분히 근사할 수 있을 만큼 다양한 형태의 분포를 표현할 수 있어야 한다.
2. **정규화**: $p_{\phi}$는 유효한 확률 분포여야 하므로, $\int p_{\phi}(\mathbf{x}) \, d\mathbf{x} = 1$을 만족해야 한다.
3. **학습 가능성**: 최적화 문제를 풀 수 있어야 한다. 앞 절에서 이야기한 것처럼 $p_{\phi}(\mathbf{x})$를 포함하는 목적 함수 $J(\phi)$를 미분해 $\nabla_{\phi} J(\phi)$를 구할 수 있어야 하고, 여기에 기댓값이 있는 경우 몬테 카를로 근사를 적용할 수 있어야 한다.
4. **샘플링 가능성**: $p_{\phi}$로부터 $\mathbf{x}$를 효율적으로 샘플링할 수 있어야 한다.

이 네 가지 조건을 동시에 만족하는 것은 쉽지 않다. 하나의 조건이 다른 조건과 상충되는 경우가 많기 때문이다. 두 번째 조건인 정규화와 네 번째 조건인 샘플링에 대해 조금 더 살펴보자.

**정규화**는 확률 분포라면 당연히 만족해야 하는 조건이지만, 복잡하면서 정규화 조건을 만족하는 분포를 만들기가 쉽지 않다. 기계 학습에서는 복잡한 함수를 만들기 위해 인공 신경망(neural network)를 활용하는데, 인공 신경망으로 정규화된 확률 분포 $p_{\phi}(\mathbf{x})$를 만드려고 시도해 보자. 먼저 $\mathbf{x}$를 입력으로 받아 실수를 내놓는 신경망 $f_{\phi}(\mathbf{x})$를 생각할 수 있다. 여기에서 매개변수 $\phi$는 신경망의 가중치 역할을 한다. 밀도함수가 되기 위해서는 음이 아닌 값이 되어야 하므로, $\exp$를 씌워 $\exp (f_{\phi}(\mathbf{x}))$로 만들자. 다음으로 정규화 조건을 만족하도록 하기 위해, 어떤 상수 $C$를 곱하자. 그럼 $C$의 값은 다음과 같이 정해진다.
$$
\int C \exp (f_{\phi}(\mathbf{x})) \;d\mathbf{x} = 1 \quad \Rightarrow \quad C = \left(\int \exp(f_{\phi}(\mathbf{x})) \;d\mathbf{x}\right)^{-1}
$$

결과적으로, 우리는 다음과 같이 정규화 조건을 만족하는 확률 분포를 얻을 수 있다.
$$
p_{\phi}(\mathbf{x}) = C \exp(f_{\phi}(\mathbf{x})) = \frac{\exp(f_{\phi}(\mathbf{x}))}{\int \exp(f_{\phi}(\mathbf{x})) \;d\mathbf{x}}
$$

$p_{\phi}(\mathbf{x})$를 잘 정의하기는 했지만, 이것을 실제로 계산하는 것은 불가능하다. 분모에 있는 적분이 너무 복잡해 계산할 수 없기 때문이다. 세 번째 조건인 학습 가능성도 문제가 된다. 분모의 적분이 $\phi$에 의존하기 때문에, $p_{\phi}(\mathbf{x})$를 $\phi$로 미분하는 것 또한 다루기 어렵다.

{{< callout type="Note" >}}
앞의 식 {{< eqref expectation-integral >}}의 복잡한 적분은 몬테 카를로로 잘 근사하지 않았나? 라는 의문이 들 수 있다. 하지만 식 {{< eqref expectation-integral >}}의 적분은 확률 분포 $p(\mathbf{x})$에 대한 기댓값이고, $p(\mathbf{x})$에서 샘플링이 가능했기 때문에 몬테 카를로 근사를 적용할 수 있었다. 반면 정규화 상수의 적분 $\int \exp(f_{\phi}(\mathbf{x})) \, d\mathbf{x}$는 우리가 샘플링할 수 있는 확률 분포에 대한 기댓값으로 자연스럽게 표현되지 않으므로, 몬테 카를로 근사를 직접 적용할 수 없다.
{{< /callout >}}


어떻게 해서 정규화 조건을 만족하면서 학습도 가능한 확률 분포 $p_{\phi}$를 얻었다고 치자. 그래도 여전히 **샘플링 가능성**이라는 네 번째 조건이 문제가 된다. 어떤 분포의 밀도 함수를 알고 있다고 해서 바로 그 분포에서 샘플링할 수 있는 것은 아니다. 밀도 함수 $p_{\phi}(\mathbf{x})$를 특정 점 $\mathbf{x}$에서 계산하는 것은 국소적인 연산이지만, 샘플링은 확률 밀도가 전체 공간에서 어떻게 분포하는지를 파악해야 하는 전역적인 문제이기 때문이다.

결국 생성 모델의 핵심 과제는 정규화와 샘플링이다. 여러 가지 생성 모델들은 이 두 문제를 다양한 접근 방식으로 해결하며, 필요에 따라 일부를 포기하기도 한다. 구체적인 생성 모델들을 알아보기 전에, 몇 가지 중요한 아이디어를 살펴보자.

### Idea 1: Autoregressive Generation

첫 번째로 살펴볼 아이디어는 autoregressive generation이다. 이 아이디어의 핵심은 다음과 같다:

{{< callout type="Idea" >}}
복잡한 확률 분포를 단순한 확률 분포들의 조합으로 분해하자. 분해된 각 분포가 충분히 단순하다면, 정규화와 샘플링 문제를 쉽게 해결할 수 있다.
{{< /callout >}}

데이터를 $\mathbf{x} = (x_{1}, \cdots, x_{L})$의 형태로 쓸 수 있다고 하자. 이때, 확률의 연쇄 법칙에 의해 확률 분포 $p_\phi({\mathbf{x}})$는 다음과 같이 $L$개의 조건부 확률 분포들의 곱으로 정확하게 표현할 수 있다. 아래 식에서 $x_{1}, \cdots, x_{i - 1}$을 $x_{1:i-1}$로 간단하게 표현했다.

$$p_{\phi}(\mathbf{x}) = \prod_{i} p_{\phi}(x_{i} \mid x_{1}, \cdots, x_{i - 1}) = \prod_{i} p_{\phi}(x_{i} \mid x_{1:i-1})$$

각각의 조건부 확률들이 정규화 조건을 만족하기만 하면 전체 확률 분포도 자동으로 정규화 조건을 만족하게 된다. 또한, 각각의 조건부 확률 분포에서 샘플링을 할 수 있다면 다음과 같이 $L$ 번의 단계를 거쳐 $\mathbf{x}$를 샘플링할 수 있다.

- $x_{1} \sim p_{\phi}(x_{1})$을 샘플링한다.
- $x_{2} \sim p_{\phi}(x_{2} \mid x_{1})$을 샘플링한다.
- $x_{3} \sim p_{\phi}(x_{3} \mid x_{1:2})$를 샘플링한다.
- $\cdots$
- $x_{L} \sim p_{\phi}(x_{L} \mid x_{1:L-1})$을 샘플링한다.

이러한 아이디어는 $L$이 고정되어 있지 않을 때에도 사용할 수 있다.

Autoregressive generation은 텍스트 생성에 특히 적합하다. 텍스트를 토큰(token) 단위로 분해하면, 각 토큰은 $K$개의 고정된 vocabulary 중 하나이다 ($K$는 보통 수만에서 수십만 정도이다). $i$번째 토큰을 $x_{i}$로 놓으면, $p_{\phi}(x_{i} \mid x_{1:i-1})$는 $K$개의 값 중 하나를 선택하는 범주형 분포(categorical distribution)가 된다. 이 분포를 구체적으로 모델링해 보자.

$f_{\phi}(x_{1:i-1})$을 이전 토큰들을 입력으로 받아 $K$차원의 실수 벡터를 출력하는 신경망이라 하고, 그 $k$번째 원소를 $f_{\phi}(x_{1:i-1})_{k}$로 표기하면 확률 분포 $p_{\phi}(x_{i} \mid x_{1:i-1})$를 다음과 같이 정의할 수 있다.

$$p_{\phi}(x_{i} = k \mid x_{1:i-1}) = \frac{\exp(f_{\phi}(x_{1:i-1})_{k})}{\sum_{j=1}^{K} \exp(f_{\phi}(x_{1:i-1})_{j})}$$

앞에서 정규화를 위해 다루기 어려운 고차원 적분이 필요했던 것과 달리, 여기에서의 분모는 $K$개 항의 유한한 합이므로 쉽게 계산할 수 있다. $\phi$에 대한 미분도 가능하므로 학습 가능성 조건도 만족한다. 샘플링 역시 $K$개의 확률값을 모두 계산한 뒤 그에 따라 하나를 선택하면 되므로 간단하다.

현재 대부분의 거대 언어 모델(LLM)은 이러한 autoregressive 생성 모델이다. Autoregressive generation은 반드시 이산적인 데이터에서만 적용할 수 있는 아이디어는 아니다. 이미지와 같은 비교적 연속적인 데이터에서도 적용하려는 시도가 예전에도{{< ref 4 >}} 최근에도{{< ref 5 >}} 이루어지고 있다.

### Idea 2: Latent Variable

두 번째로 살펴볼 아이디어는 잠재 변수(latent variable)의 도입이다.

{{< callout type="Idea" >}}
데이터 $\mathbf{x}$를 직접 모델링하는 대신, 데이터의 생성 과정을 설명하는 숨겨진 변수 $\mathbf{z}$를 도입하자. $\mathbf{z}$의 분포와 $\mathbf{z}$가 주어졌을 때 $\mathbf{x}$의 분포를 각각 단순하게 설정하면, 정규화와 샘플링을 쉽게 해결할 수 있다.
{{< /callout >}}

잠재 변수 $\mathbf{z}$는 직접 관측할 수 없지만, 데이터의 생성 과정을 설명하는 변수이다. 예를 들어, 사람의 얼굴 이미지를 생성하는 문제에서 잠재 변수는 자세, 표정, 조명 등 이미지의 특성을 결정하는 요인들에 해당할 수 있다. 잠재 변수를 도입하면, 데이터의 확률 분포를 다음과 같이 표현할 수 있다.

$$p_{\phi}(\mathbf{x}) = \int p_{\phi}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \int p_{\phi}(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

여기에서 $p(\mathbf{z})$는 잠재 변수의 사전 분포(prior distribution)이고, $p_{\phi}(\mathbf{x} \mid \mathbf{z})$는 잠재 변수가 주어졌을 때 데이터의 조건부 분포이다. 이 두 분포는 일반적으로 단순한 분포로 정의한다. 예를 들어, $p(\mathbf{z})$를 표준 정규 분포로, $p_{\phi}(\mathbf{x} \mid \mathbf{z})$를 $\mathbf{z}$에 의해 평균과 분산이 결정되는 정규 분포로 설정할 수 있다. 즉, $\boldsymbol{\mu}_{\phi}(\mathbf{z})$와 $\boldsymbol{\Sigma}_{\phi}(\mathbf{z})$가 결정론적인 함수일 때, 다음과 같이 정의할 수 있다.

$$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad p_{\phi}(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_{\phi}(\mathbf{z}), \boldsymbol{\Sigma}_{\phi}(\mathbf{z}))$$

두 분포를 이렇게 단순하게 설정하면 표현력이 부족하지 않을까라는 의문이 들 수 있다. 하지만 $p_{\phi}(\mathbf{x} \mid \mathbf{z})$의 평균 과 분산이 $\mathbf{z}$에 따라 달라지도록 신경망으로 모델링하면, $\mathbf{z}$마다 서로 다른 단순한 분포가 만들어진다. 이를 $p(\mathbf{z})$에 대해 적분하면 무수히 많은 단순한 분포가 합쳐져 $p_{\phi}(\mathbf{x})$라는 복잡한 분포를 표현할 수 있게 된다.

이 방법이 정규화와 샘플링 문제를 어떻게 해결하는지 살펴보자. 먼저 정규화를 살펴보면, $p(\mathbf{z})$와 $p_{\phi}(\mathbf{x} \mid \mathbf{z})$가 각각 유효한 확률 분포이기만 하면, $p_{\phi}(\mathbf{x})$도 자동으로 정규화 조건을 만족한다. 증명은 다음과 같이 할 수 있다.

$$\int p_{\phi}(\mathbf{x}) \, d\mathbf{x} = \int \left( \int p_{\phi}(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z} \right) d\mathbf{x} = \int p(\mathbf{z}) \underbrace{\left( \int p_{\phi}(\mathbf{x} \mid \mathbf{z}) \, d\mathbf{x} \right)}_{= 1} d\mathbf{z} = \int p(\mathbf{z}) \, d\mathbf{z} = 1$$

샘플링 역시 간단하다. 다음과 같이 두 단계를 거쳐 $\mathbf{x}$를 생성할 수 있다.

- $\mathbf{z} \sim p(\mathbf{z})$에서 잠재 변수를 샘플링한다.
- $\mathbf{x} \sim p_{\phi}(\mathbf{x} \mid \mathbf{z})$에서 데이터를 샘플링한다.

각 단계에서 샘플링하는 분포가 단순하므로, 전체 과정도 효율적이다.

하지만 학습에는 어려움이 있다. 확률 밀도 $p_{\phi}(\mathbf{x})$가 $p(\mathbf{z})$와 $p_{\phi}(\mathbf{x} \mid \mathbf{z})$가 연관된 복잡한 적분으로 표현되기 때문이다. 학습의 어려움에 대해서는 대표적인 latent variable 모델인 variational autoencoder (VAE)를 다룰 때 더 자세히 살펴보자.

샘플링 과정을 더 단순화할 수도 있다. 두 번째 단계에서 $\mathbf{x}$를 $p_{\phi}(\mathbf{x} \mid \mathbf{z})$로부터 확률적으로 샘플링하는 대신, 신경망 $G_{\phi}$를 이용한 결정론적인 변환 $\mathbf{x} = G_{\phi}(\mathbf{z})$로 대체하는 것이다. 이렇게 하면 확률 밀도 $p_{\phi}(\mathbf{x})$를 다루기가 더 어려워지기 때문에, 이를 해결하기 위한 아이디어가 필요하다. 이러한 방식은 generative adversarial network (GAN)나 normalizing flow (NF)를 다룰 때 더 자세히 살펴보자.

### Idea 3: 정규화 없이 샘플링하기

세 번째로 살펴볼 아이디어는 정규화를 포기하는 것이다.

{{< callout type="Idea" >}}
정규화된 밀도 함수 $p_{\phi}(\mathbf{x})$를 명시적으로 구하지 않고도 원하는 분포에서 샘플링할 수 있는 방법을 찾자.
{{< /callout >}}

놀랍게도 이러한 방법이 존재한다. 바로 무작위 알고리즘 중 하나인 Markov chain Monte Carlo (MCMC) 를 이용하는 것이다. 이러한 방식은 score-based model을 다룰 때 더 자세히 살펴보자.

## 정리

이 글에서는 생성 모델을 다루는 문제를 수학적으로 모델링하는 방법을 살펴보았다. 먼저 매개변수화된 분포를 도입해, 생성 문제를 실제 데이터의 확률 분포 $p_{\mathrm{data}}$에 가까운 분포 $p_{\phi}$를 찾고 이 분포에서 샘플링하는 문제로 모델링하였다. 이 문제는 매개변수 $\phi$에 대한 최적화 문제로 바꾸어 해결할 수 있다.

이 최적화 문제의 목적 함수는 $\phi$로 미분한 결과를 구할 수 있어야 한다. 만약 기댓값이 포함되어 있다면, 몬테 카를로 근사를 적용할 수 있어야 한다.

$p_{\phi}$를 모델링할 때는 표현력, 정규화, 학습 가능성, 샘플링 가능성이라는 네 가지 조건을 고려해야 하며, 특히 정규화와 샘플링 가능성이 가장 큰 문제이다. 이를 해결하기 위한 세 가지 아이디어를 살펴보았다.

지금부터는 다양한 생성 모델들을 살펴보면서 이 아이디어들이 실제로 어떻게 적용되는지 알아보자.
1. Autoregressive models
2. Variational Autoencoder (VAE)
3. Generative Adversarial Network (GAN)
4. Score-based models
5. Normalizing Flow (NF)
6. Diffusion models

{{< reflist >}}
{{< refitem 1 >}}Lai, Chieh-Hsin and Song, Yang and Kim, Dongjun and Mitsufuji, Yuki and Ermon, Stefano. [*The principles of diffusion models*](https://arxiv.org/abs/2510.21890). arXiv preprint, 2025.{{< /refitem >}}
{{< refitem 2 >}}Wikipedia, [*Maximum likelihood estimation*](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).{{< /refitem >}}
{{< refitem 3 >}}Wikipedia, [*Maximum a posteriori estimation*](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation).{{< /refitem >}}
{{< refitem 4 >}}van den Oord, Aaron and Kalchbrenner, Nal and Kavukcuoglu, Koray. [*Pixel Recurrent Neural Networks*](https://arxiv.org/abs/1601.06759). arXiv preprint, 2016.{{< /refitem >}}
{{< refitem 5 >}}Li, Tianhong and Tian, Yonglong and Li, He and Deng, Mingyang and He, Kaiming. [*Autoregressive Image Generation without Vector Quantization*](https://arxiv.org/abs/2406.11838). NeurIPS, 2024.{{< /refitem >}}
{{< /reflist >}}
