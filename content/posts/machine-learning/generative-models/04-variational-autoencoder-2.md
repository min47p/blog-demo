---
title: "Generative Models - 04. Variational Autoencoder (2)"
date: 2026-02-14
tags: ["machine-learning", "generative-models"]
draft: false
---

*이 글은 Claude Opus 4.6의 도움을 받아 작성했다.*

이전 포스트에서 posterior를 해석적으로 구하기 어려울 때 활용할 수 있는 베이지안 추론 방법인 MCMC와 variational Bayes를 살펴보았다. 하지만 이 방법들은 딥러닝에 적용하기 어렵다. 그 이유를 정리하면 다음과 같다.

**문제점 1. 각 관측 샘플마다 다른 posterior를 얻어야 한다**. $N$개의 샘플 $\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$이 주어졌을 때, 이들로부터 얻어지는 $\mathbf{z}$의 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(1)})$, $\cdots$, $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(N)})$은 모두 다른 분포이다. MCMC를 사용하든 variational Bayes든, 이 분포들을 일일히 다르게 취급해야 한다. 딥러닝 모델을 학습시키기 위해서는 수많은 샘플이 필요하므로, 이는 치명적인 문제점이다.

**문제점 2. 높은 분산 문제**. 이 문제는 variational Bayes에서 발생했다. $q_{\phi}$의 매개변수 $\phi$를 최적화할 때 log-derivative trick과 몬테 카를로 근사를 사용하면 추정량의 분산이 높아 제대로 된 근사가 이루어지지 않는다 (매우 많은 샘플을 활용해야 제대로 근사할 수 있다). 이 문제는 이전 포스트에서 자세히 살펴보지 않고 넘어갔다 ([참고](../03-variational-autoencoder-1/#eq-high-variance-of-mc)).

**문제점 3. 샘플링 효율성**. 이 문제는 MCMC에서 발생했다. MH algorithm가 posterior를 의미 있는 수준으로 근사하기 위해서는 Markov chain을 충분히 길게 따라가야 한다. 이렇게 얻은 샘플이 한두 개 필요한 것이 아니다.

세 가지 문제점으로 인해, 이 방법들로 딥러닝 모델을 학습시키면 극도로 비효율적이다. 이번 포스트에서는 이 문제점들을 어떻게 해결하는지 살펴볼 것이다. 우선 MCMC를 포기하고 variational Bayes의 접근 방법을 채택해, posterior를 매개화된 확률 분포 $q_{\phi}$로 근사할 것이다. 

# Amortized Inference

문제점 1을 해결하는 방법은 간단하다. 지금까지는 $\mathbf{z}$의 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(1)})$, $\cdots$, $p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(N)})$를 제각각 근사해야 했다. 즉, $N$개의 최적화 문제
$$
q_{\phi^{(1)}}(\mathbf{z}) \approx p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(1)}), \quad \cdots, \quad q_{\phi^{(N)}}(\mathbf{z}) \approx p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(N)})
$$
를 각각 해결해 매개변수 $\phi^{(1)}$, $\cdots$, $\phi^{(N)}$를 얻어야 했다.

이 문제를 한번에 해결할 수는 없을까? 즉, 하나의 $\phi$를 사용해 $N$개의 posterior를 모두 근사하는 것이다. 그렇게 되면 각각의 근사된 posterior는 제공되는 관측 데이터의 샘플에 의존해야 하기 때문에, 지금까지 사용했던 $q_{\phi}(\mathbf{z})$ 대신 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$라는 표기법을 사용해야 할 것이다. 이제 다음과 같은 문제로 바뀐다.
$$
q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(1)}) \approx p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(1)}), \quad \cdots, \quad q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(N)}) \approx p_{\theta}(\mathbf{z} \mid \mathbf{x}^{(N)})
$$

이렇게 하나의 매개변수로 모든 posterior를 동시에 근사할 수 있는 이유는 우리의 모델에서 $\mathbf{x}$와 $\mathbf{z}$가 긴밀하게 연관되어 있기 때문이다. 다만 이것이 실제로 작동하려면, $q_{\phi}$가 이 복잡한 관계를 표현할 수 있을 만큼 충분히 유연해야 한다. 전통적인 통계적 추론에서 사용하는 모델로는 이것이 어렵지만, 딥러닝에서는 표현력이 높은 신경망을 활용해 모델링하므로 이것이 가능하다. 단, 공유된 $\phi$는 개별 $\phi^{(i)}$보다 유연성이 떨어지므로 각 샘플에 대한 근사의 정밀도는 희생될 수 있다. 이러한 접근 방법을 **amortized inference**라 한다.

이제 $p_{\theta}$와 $q_{\phi}$의 의미가 더 명확해진다. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$는 잠재 변수 $\mathbf{z}$로부터 관측 데이터 $\mathbf{x}$를 생성하는 역할을 하므로 **decoder**라 부르고, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$는 관측 데이터 $\mathbf{x}$로부터 잠재 변수 $\mathbf{z}$의 분포를 추론하는 역할을 하므로 **encoder**라 부른다. 이는 입력을 저차원 표현으로 압축하는 encoder와 이를 다시 복원하는 decoder로 구성된 autoencoder와 유사한 구조이며, variational autoencoder라는 이름도 여기에서 나왔다. 다만, 이 포스트에서는 $q_{\phi}$의 의미로 encoder보다는 posterior의 근사 분포임을 더 강조하고자 한다. 그래서 이후로는 encoder/decoder라는 용어를 사용하지 않았다.

## 근사 분포의 모델링

Posterior의 근사 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 모델링할 때 자주 사용되는 방법은 $p_{\theta}$를 모델링할 때와 비슷하다. $\mathbf{x}$를 입력으로 받고 매개변수 $\phi$를 가중치로 하는 신경망 $\boldsymbol{\mu}_{\phi}(\mathbf{x})$와 $\boldsymbol{\Sigma}_{\phi}(\mathbf{x})$를 정의한 다음, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 다음과 같이 정의한다. 식을 간결하게 적기 위해 $p_{\theta}$의 모델링에서와 똑같은 기호 $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$를 사용하고, 아래첨자($\theta$ 또는 $\phi$)와 입력($\mathbf{z}$ 또는 $\mathbf{x}$)으로 구분하겠다.

{{< eqlabel normal-posterior >}}
$$q_{\phi}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z};\, \boldsymbol{\mu}_{\phi}(\mathbf{x}), \boldsymbol{\Sigma}_{\phi}(\mathbf{x}))$$

한편, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 이렇게 단순한 정규분포로 설정하는 것에 대해 의문이 생길 수 있다. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 단순했던 것은 우리의 모델링 가정이니 넘어가더라도, $\mathbf{z}$의 posterior는 모델링에 의해 결정되는 분포인데 이것을 우리 마음대로 단순하게 근사해도 되는 걸까?

어떤 연속 확률 분포에서 밀도함수의 값이 극대가 되는 지점을 mode라고 한다 (엄밀한 정의는 아니고, 관습으로 사용되는 용어이다). 실제 posterior에는 mode가 여러 개 있을 수 있는데, 정규 분포는 하나의 mode만 가진다. 그래서 정규 분포로 정의된 $q_{\phi}$로 posterior를 근사하게 되면 여러 mode 중 하나에만 집중하고 나머지는 무시하거나, 모든 mode에 걸쳐 있는 분포를 갖게 된다. Variational Bayes에서는 전자의 현상이 일어나는데, 자세한 설명은 아래를 참고하자.

{{< toggle title="Variational Bayes에서의 mode-seeking behavior" >}}
[이전 포스트](../03-variational-autoencoder-1/#variational-bayes)에서 우리는 미지의 분포 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 매개화된 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$로 근사하기 위해 KL divergence를 최소화하고자 할 때, 미지의 분포를 왼쪽에 놓아야 할지 오른쪽에 놓아야 할지 논의했다. 이 선택이 $q_{\phi}$의 behavior를 결정한다.

표기법을 간단히 해서 미지의 분포를 $p(\mathbf{x})$, 이를 근사하기 위한 분포를 $q(\mathbf{x})$라고 하자. $p$는 mode가 여러 개인 분포이고, $q$는 mode가 1개인 분포라고 하자. 먼저 $p$가 KL divergence의 왼쪽에 있는 경우를 살펴보자.
$$
D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})} \left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]
$$
이 식에서, $p(\mathbf{x}) > 0$인 점에서 $q(\mathbf{x}) \to 0$이면 $\log (p(\mathbf{x})/q(\mathbf{x})) \to \infty$이므로 KL divergence가 발산한다. 따라서 KL divergence를 최소화하면 $q$는 $p$가 유의미한 값을 가지는 모든 영역을 빠짐없이 덮도록 최적화된다. $q$는 mode가 1개이므로, $p$의 모든 mode를 포함하게 된다. 이때 $q$의 mode는 $p$의 모든 mode를 평균 낸 것과 비슷한 역할을 한다. 이러한 현상을 **mean-seeking** 또는 **mode-covering**이라고 한다.

다음으로, $p$가 KL divergence의 오른쪽에 있는 경우를 살펴보자.
$$
D_{\mathrm{KL}}(q \| p) = \mathbb{E}_{\mathbf{x} \sim q(\mathbf{x})} \left[\log \frac{q(\mathbf{x})}{p(\mathbf{x})}\right]
$$
이번에는 반대로 $q(\mathbf{x}) > 0$인 점에서 $p(\mathbf{x}) \to 0$이면 발산한다. 따라서 $q$는 $p$가 작은 영역을 피하려고 하고, 결과적으로 $p$의 여러 mode 중 하나에 집중하게 된다. 이러한 현상을 **mode-seeking**이라고 한다.

Variational Bayes의 목적 함수를 결정할 때, 우리는 계산 가능성을 기준으로 후자($D_{\mathrm{KL}}(q \| p)$)를 택했다. 따라서 $q_{\phi}$는 실제 posterior의 mode 중 하나에 집중하고 나머지는 무시하는 경향을 가진다. 이 현상의 근본적인 원인은 KL divergence의 비대칭성이다.
{{< /toggle >}}

이렇게 $q_{\phi}$가 실제 posterior를 제대로 근사하지 못하는 것은 모델링의 한계이다. 그래도 지금은 가장 기본적인 형태인 이 모델링을 사용하자.

## Amortized Inference에서의 목적 함수

Variational Bayes에서 매개변수 $\theta$와 하나의 샘플 $\mathbf{x}$를 고정했을 때, $q_{\phi}(\mathbf{z})$의 매개변수 $\phi$를 구하기 위해 최대화해야 하는 목적 함수는 다음과 같이 정의된 ELBO였다.
$$
\mathcal{L}(\phi; \mathbf{x}, \theta) = \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})} [\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z})]
$$
$$
\phi = \argmax_{\phi} \mathcal{L}(\phi; \mathbf{x}, \theta)
$$

마찬가지로 amortized inference를 적용한 variational bayes에서 매개변수 $\theta$를 고정했을 때, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$의 매개변수 $\phi$를 구하기 위해 최대화해야 하는 목적 함수는 다음과 같다. 모든 샘플에 대해 동시에 최적화하는 전략은 ELBO의 기댓값으로 표현된다.
$$
\begin{align*}
\phi &= \argmax_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\mathcal{L}(\phi; \mathbf{x}, \theta)]
\end{align*}
$$

목적 함수를 이용해 $N$ 개의 관측 데이터 샘플을 이용해 몬테 카를로 근사하면 다음과 같다.
$$
\phi \approx \argmax_{\phi} \frac{1}{N} \sum_{n=1}^{N} \mathcal{L}(\phi; \mathbf{x}^{(n)}, \theta)
$$

# Wake-Sleep Algorithm

지금까지 우리의 논리를 되짚어 보자. 우리는 생성 모델의 목표를 $p_{\mathrm{data}}(\mathbf{x})$와 $p_{\theta}(\mathbf{x})$ 간의 KL divergence를 최소화하는 것으로 설정했다. 모델로는 latent variable model을 도입했다. 이 모델에서 샘플 $\mathbf{x}$의 marginal likelihood $p_{\theta}(\mathbf{x})$를 직접 계산하기 어렵기 때문에, posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 통해 이를 우회하고자 했다. Posterior를 다루기 위해 베이지안 추론을 살펴보았고, 이 중 posterior를 매개화된 분포 $q_{\phi}(\mathbf{z})$로 근사하는 variational Bayes를 채택했다. 이때, 각 샘플마다 다른 posterior를 얻는 것은 비효율적이므로, 하나의 매개변수로 모든 posterior를 근사하는 amortized inference 방식을 도입했다.

이제 VAE로 넘어갈 수 있는 준비를 갖추었다. 그 전에, VAE 이전에 존재하던 생성 모델의 학습 알고리즘인 wake-sleep algorithm을 살펴보자. 딥러닝의 아버지라고 불리는 G. E. Hinton을 포함한 딥러닝의 개척자들이 1995년 제안한 이 알고리즘은{{< ref 2 >}}, 딥러닝이라는 용어도 없었던 초창기의 연구이다. 그래서 논문에서 사용하는 용어나 개념이 지금 흔히 쓰이는 것들과 다르다. 특히 분포 간의 divergence를 최소화하는 통계학의 관점보다는 주어진 데이터를 가장 짧게 설명하는 모델을 찾는다는 정보 이론의 관점에서 접근하고 있다. 이러한 관점을 **minimum description length (MDL)** 이라고 한다. 여기에 적은 내용은 (Claude의 도움을 받아) 우리의 관점으로 바꿔 서술한 것이다. MDL에 대한 설명은 이 포스트에서 자세히 다루지 않을 예정이므로 toggle box 안에 적었다.

{{< toggle title="Minimum Description Length" >}}
주어진 데이터 $\mathbf{x}$를 설명한다는 것은 $\mathbf{x}$를 어떤 메시지로 encoding한다는 뜻이다. 이 메시지를 decoding해 원래의 $\mathbf{x}$를 복원할 수 있어야 한다.

구체적인 설명을 위해 두 명의 사람, sender와 receiver가 있다고 하자. 편의상 $\mathbf{x}$가 이산 확률 변수라고 하자. (연속 확률 변수일 경우 양자화를 통해 이산 확률 변수로 만들 수 있다.) 두 사람은 $\mathbf{x}$에 대한 확률 분포 $p(\mathbf{x})$를 서로 공유하고 있다. 두 사람의 목표는 $\mathbf{x}$의 값을 최대한 효율적으로 주고받는 것이다. 구체적인 상황은 다음과 같다.
1. Sender는 $\mathbf{x}$를 $p(\mathbf{x})$에서 샘플링한 뒤, 이를 0과 1로 이루어진 **binary string (메시지)** 으로 encoding해 receiver에게 전송한다.
2. Receiver는 전송받은 binary string (메시지)을 다시 decoding해 $\mathbf{x}$로 정확하게 복원해야 한다.

이러한 상황에서, sender가 전송하는 메시지의 길이의 기댓값을 최소화하고자 한다. 두 사람은 모두 $p(\mathbf{x})$가 어떤 분포인지 정확히 알고 있고, $\mathbf{x}$를 어떻게 encoding하고 decoding할지 사전에 합의할 수 있다. Sender는 $\mathbf{x}$의 값을 최대한 짧은 길이의 메시지로 설명해야 하기 때문에, 이 문제를 minimum description length라고 한다.

Shannon에 의하면, 메시지의 길이의 기댓값의 이론적인 최솟값은 확률 분포 $p(\mathbf{x})$에 의해 정해지며, 다음과 같다. 이를 $p(\mathbf{x})$의 Shannon entropy라고 부른다.
$$
H(p) = \mathbb{E}_{\mathbf{x} \sim p(\mathbf{x})}[-\log_{2} p(\mathbf{x})]
$$
위 식의 의미는 어떤 $\mathbf{x}$가 샘플링될 확률이 $p(\mathbf{x})$일 때, $\mathbf{x}$를 길이 $-\log_{2} p(\mathbf{x})$의 메시지로 encoding해야 한다는 의미이다. 실제로, Huffman coding 등 최적의 encoding 알고리즘을 사용하면 어떤 확률 분포 $p(\mathbf{x})$가 주어졌을 때 $\mathbf{x}$가 encoding되는 메시지의 길이는 대략 $-\log_{2} p(\mathbf{x})$ 가 된다.

여기에서는 구체적인 encoding 과정에 대한 자세한 설명은 생략하고, '확률 분포 $p(\mathbf{x})$를 이용해 $\mathbf{x}$를 encoding한다'라고 표현하겠다. 이 문장은 (최적의 알고리즘을 사용해) $\mathbf{x}$를 길이 $-\log p(\mathbf{x})$로 encoding하겠다는 의미이다. 로그의 밑이 달라지면 값이 상수 배 차이날 뿐이므로, 로그의 밑을 생략해서 나타낸다.
{{< /toggle >}}

Hinton et al.이 생성 모델을 설계하고자 할 때 겪은 문제는 학습 데이터를 설명해 줄 teacher가 없어 학습에 필요한 적절한 signal을 만들어낼 수 없다는 것이었다. 이를 해결하기 위해 $\mathbf{z}$가 주어질 때 $\mathbf{x}$의 분포를 나타내는 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$와, $\mathbf{x}$가 주어질 때 $\mathbf{z}$의 분포를 나타내는 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 각각 설계해 서로를 학습에 활용하고자 했다.

알고리즘은 두 개의 phase를 반복한다. Wake phase에서는 $\phi$를 고정하고, 실제 데이터 $\mathbf{x}$와 $q_{\phi}$에서 샘플링한 $\mathbf{z}$를 이용해 $\theta$를 학습시킨다. Sleep phase에서는 반대로 $\theta$를 고정하고, $p_{\theta}$에서 샘플링한 가상의 데이터 $\tilde{\mathbf{x}}$를 이용해 $\phi$를 학습시킨다. 알고리즘의 이름은 실제 세계의 데이터를 보며 학습하는 것을 wake에, 실제 데이터를 보지 않고 상상하며 학습하는 것을 sleep에 비유해 지어진 것이다.

## Wake Phase

Wake phase에서는 $\phi$를 고정한 뒤 실제 관측 데이터의 샘플을 이용해 $\theta$를 학습시킨다. 먼저, 하나의 샘플에 대한 목적 함수는 다음과 같다.

{{< eqlabel description-cost >}}
$$
C(\mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x}, \mathbf{z}) + \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
$$

MDL의 관점에서 이 목적 함수는 모델이 $\mathbf{x}$를 설명하는 데 필요한 비용(메시지의 길이)를 의미한다. 자세한 설명은 아래를 참고하자.

{{< toggle title="하나의 샘플에 대한 목적 함수 유도 (MDL)" >}}
MDL의 관점에서, 목적 함수는 '샘플을 전송하기 위해 필요한 메시지의 평균 길이'로 정의해야 한다.

$p_{\theta}$와 $q_{\phi}$가 주어진 상황에서, sender가 하나의 샘플 $\mathbf{x}$를 전송하는 방법은 다음과 같다.
1. 임의의 $\mathbf{z}$를 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$에서 샘플링한다.
2. $p_{\theta}(\mathbf{z})$를 이용해 $\mathbf{z}$를 길이 $-\log p_{\theta}(\mathbf{z})$의 메시지로 encoding한다.
3. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 이용해 $\mathbf{x}$를 길이 $-\log p_{\theta}(\mathbf{x} \mid \mathbf{z})$의 메시지로 encoding한다.

이때 총 메시지의 길이는 $-\log p_{\theta}(\mathbf{z}) - \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) = -\log p_{\theta}(\mathbf{x}, \mathbf{z})$가 된다. 참고로, 단계 2에서 $\mathbf{z}$를 encoding할 때 $\mathbf{z}$를 샘플링한 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 아니라 $p_{\theta}(\mathbf{z})$를 이용해야 하는 이유는, receiver는 이 시점에서 $\mathbf{x}$를 알지 못해 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 이용한 decoding을 수행할 수 없기 때문이다. Receiver의 decoding 과정은 다음과 같다.

- $p_{\theta}(\mathbf{z})$를 이용해 단계 2의 메시지를 decoding해 $\mathbf{z}$를 복원한다.
- $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 이용해 단계 3의 메시지를 decoding해 $\mathbf{x}$를 복원한다.

하나의 샘플 $\mathbf{x}$를 전송하기 위한 메시지의 길이의 기댓값은 다음과 같다.
{{< eqlabel single-data-cost >}}
$$
\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x}, \mathbf{z})]
$$

그런데 단계 1에서 $\mathbf{z}$를 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$에서 자유롭게 샘플링했음에 주목하자. 자유롭게 샘플링했다는 것은 여기에 **$\mathbf{x}$와 독립인 추가적인 확률 변수에 대한 정보를 심을 수 있다**는 것을 뜻한다. 이것이 무슨 의미인지 구체적으로 살펴보자.

분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$에 대한 최적의 encoding/decoding 알고리즘을 생각하면, 각 $\mathbf{z}$는 길이 $-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$의 메시지로 encoding된다. 이제 모든 자리가 $1/2$의 확률로 0이고 $1/2$의 확률로 1인 (따라서 $\mathbf{x}$와도 독립인) binary string $m$이 있다고 하자. 위 decoding 알고리즘을 $m$에 적용하면, $m$의 prefix에 대응되는 어떤 $\mathbf{z}_m$을 얻을 수 있다. 이때, $\mathbf{z}_{m}$은 다음 두 가지 성질을 가지고 있다.

1. $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 이용해 $\mathbf{z}_m$을 encoding하면 $m$의 prefix가 복원된다. Encoding과 decoding은 역함수 관계이기 때문이다. 이 prefix의 길이는 $-\log q_{\phi}(\mathbf{z}_m \mid \mathbf{x})$이다.
2. $\mathbf{z}_m$의 분포는 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$와 동일하다. $\mathbf{z}_{m}$을 encoding한 메시지 길이는 $-\log q_{\phi}(\mathbf{z}_{m} \mid \mathbf{x})$이다. $m$이 균일 분포이므로, $m$이 이 메시지와 일치하는 prefix를 가질 확률은 $(1/2)^{-\log q_{\phi}(d\mathbf{z} \mid \mathbf{x})} = q_{\phi}(\mathbf{z} \mid \mathbf{x})$이다.

따라서, 이렇게 얻은 $\mathbf{z}_{m}$을 이용하면 $m$의 $-\log q_{\phi}(\mathbf{z}_m \mid \mathbf{x})$ 개의 비트를 $\mathbf{x}$와 함께 전송할 수 있다!

Sender가 $\mathbf{x}$와 함께 $m$의 첫 $-\log q_{\phi}(\mathbf{z}_m \mid \mathbf{x})$ 비트를 전송하는 방법은 다음과 같다.
1. 위에서 설명한 $\mathbf{z}_m$을 구한다. $q_{\phi}(\mathbf{z} \mid \mathbf{x})$을 이용해 $\mathbf{z}_{m}$을 encoding하면 $m$의 prefix가 나와야 한다.
2. $p_{\theta}(\mathbf{z})$를 이용해 $\mathbf{z}$를 길이 $-\log p_{\theta}(\mathbf{z})$의 메시지로 encoding한다.
3. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 이용해 $\mathbf{x}$를 길이 $-\log p_{\theta}(\mathbf{x} \mid \mathbf{z})$의 메시지로 encoding한다.

Receiver의 입장에서는, 다음과 같은 순서로 $\mathbf{x}$ 뿐만 아니라 $m$의 첫 $-\log q_{\phi}(\mathbf{z}_m \mid \mathbf{x})$ 비트를 알아낼 수 있다. 단계 1과 2는 동일하고, 단계 3이 추가되었다.
1. $p_{\theta}(\mathbf{z})$를 이용해 단계 2의 메시지를 decoding해 $\mathbf{z}$를 복원한다.
2. $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 이용해 단계 3의 메시지를 decoding해 $\mathbf{x}$를 복원한다.
3. 이제 $\mathbf{x}$와 $\mathbf{z}$를 모두 알고 있으니, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 이용해 $\mathbf{z}$를 encoding해 $m$의 첫 $-\log q_{\phi}(\mathbf{z}_m \mid \mathbf{x})$ 비트를 구한다.

이렇게 $\mathbf{z}$의 샘플링에 추가 정보를 심어 비트를 되돌려 받을 수 있다{{< ref 4 >}}. 되돌려 받을 수 있는 메시지의 길이의 기댓값은 다음과 같으며, 다름 아닌 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$의 Shannon entropy이다.
{{< eqlabel bits-back-cost >}}
$$
\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
$$

이제 $N$ 개의 IID 샘플 $\mathbf{x}^{(1)}, \cdots, \mathbf{x}^{(N)}$을 동시에 전송하는 상황을 생각해 보자. 각 샘플이 독립이므로, $n$번째 샘플의 $\mathbf{z}^{(n)}$ 샘플링에 $(n + 1)$번째 샘플의 메시지 일부를 심을 수 있다. 이렇게 하면 전체 메시지의 길이가 줄어든다. 구체적인 알고리즘은 서술이 복잡해서 아래 toggle box에 숨겼다.

{{< toggle title="$N$ 개의 샘플을 동시에 전송하는 알고리즘" >}}
비트를 넣을 수 있는 스택 $B$를 도입하자. $B$에는 비트를 넣거나(push) 꺼낼(pop) 수 있으며, 마지막에 넣은 비트가 먼저 나온다.

Sender는 $B$를 빈 상태로 초기화한 뒤, $n = N, N - 1, \cdots, 1$ 순서로 다음을 반복한다.

1. $B$가 비어 있지 않으면, $B$의 비트들을 $m$으로 사용하여 $\mathbf{z}^{(n)}$을 결정한다. 이 과정에서 $B$의 약 $-\log q_{\phi}(\mathbf{z}^{(n)} \mid \mathbf{x}^{(n)})$ 개의 비트가 소비된다. $B$가 비어 있으면 $\mathbf{z}^{(n)}$을 $q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(n)})$에서 자유롭게 샘플링한다.
2. $p_{\theta}(\mathbf{x} \mid \mathbf{z}^{(n)})$를 이용해 $\mathbf{x}^{(n)}$을 encoding하여 얻은 $-\log p_{\theta}(\mathbf{x}^{(n)} \mid \mathbf{z}^{(n)})$ 개의 비트를 $B$에 넣는다.
3. $p_{\theta}(\mathbf{z})$를 이용해 $\mathbf{z}^{(n)}$을 encoding해 얻은 $-\log p_{\theta}(\mathbf{z}^{(n)})$ 개의 비트를 $B$에 넣는다.

모든 샘플을 처리한 후, $B$에 남아 있는 비트를 receiver에게 전송한다.

Receiver는 sender가 수행한 과정들을 거꾸로 수행하면 된다. 구체적으로는, 전송받은 비트를 모두 $B$에 넣고 $n = 1, 2, \cdots, N$ 순서로 다음을 반복한다.

1. $B$에서 비트를 꺼내, $p_{\theta}(\mathbf{z})$를 이용해 decoding하여 $\mathbf{z}^{(n)}$을 얻는다.
2. $B$에서 비트를 꺼내, $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 이용해 decoding하여 $\mathbf{x}^{(n)}$을 얻는다.
2. $q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(n)})$를 이용해 $\mathbf{z}^{(n)}$을 encoding하여 나온 비트를 $B$에 넣는다.

Sender의 각 단계에서 $B$에 $-\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}^{(n)})$ 비트가 추가되고, $n = N$을 제외하면 $-\log q_{\phi}(\mathbf{z}^{(n)} \mid \mathbf{x}^{(n)})$ 비트가 소비된다. 따라서, $N$개 샘플의 총 전송 비용 ($B$에 남는 비트 수)은 다음과 같다.

$$
\sum_{n=1}^{N} [-\log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}^{(n)})] - \sum_{n=1}^{N-1} [-\log q_{\phi}(\mathbf{z}^{(n)} \mid \mathbf{x}^{(n)})]
$$
{{< /toggle >}}

$N \to \infty$일 때, 하나의 샘플을 전송하기 위해 필요한 메시지의 평균 길이는 아래 값으로 수렴한다. 식 {{< eqref single-data-cost >}}에서 {{< eqref bits-back-cost >}}를 뺀 결과이다.
$$
\mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x}, \mathbf{z}) + \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
$$

따라서 하나의 샘플 $\mathbf{x}$를 설명하는 데 필요한 비용을 식 {{< eqref description-cost >}}과 같이 정의할 수 있다.
{{< /toggle >}}

이제 wake phase의 목적 함수를 다음과 같이 쓸 수 있다.

{{< eqlabel wake-phase >}}
$$
\begin{align*}
\theta &= \argmin_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [C(\mathbf{x})]\\
&= \argmin_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x}, \mathbf{z}) + \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
\end{align*}
$$

참고로, 두 번째 항인 $\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$는 $\theta$에 무관한 항이므로 무시해도 좋다. 하지만 MDL에 의한 유도에서 자연스럽게 나오는 항이기도 하고, 이어지는 논의에서 중요한 역할을 하기 때문에 생략하지 않았다.

Wake phase의 목적 함수를 divergence 최소화 관점에서 이해해 보자. 먼저, $C(\mathbf{x})$에 $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{z} \mid \mathbf{x}) p_{\theta}(\mathbf{x})$를 대입하면 다음과 같다.

$$
\begin{align*}
C(\mathbf{x}) &= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x}, \mathbf{z}) + \log q_{\phi}(\mathbf{z} \mid \mathbf{x})] \\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\log q_{\phi}(\mathbf{z} \mid \mathbf{x}) - \log p_{\theta}(\mathbf{z} \mid \mathbf{x}) - \log p_{\theta}(\mathbf{x})] \\
&= D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})) - \log p_{\theta}(\mathbf{x})
\end{align*}
$$

두 번째 항인 $-\log p_{\theta}(\mathbf{x})$의 기댓값도 다음과 같이 KL divergence로 나타낼 수 있다.

$$
\begin{align*}
\mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [-\log p_{\theta}(\mathbf{x})]
&= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[\log \frac{p_{\mathrm{data}}(\mathbf{x})}{p_{\theta}(\mathbf{x})}\right] + \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [-\log p_{\mathrm{data}}(\mathbf{x})] \\
&= D_{\mathrm{KL}}(p_{\mathrm{data}}(\mathbf{x}) \| p_{\theta}(\mathbf{x})) + \mathrm{const.}
\end{align*}
$$

위 식의 두 번째 항은 데이터 분포에만 의존하고 $\theta$와 $\phi$에는 무관한 상수이므로, 이를 $\mathrm{const.}$로 표시했다. 이제 전체 목적 함수를 다음과 같이 쓸 수 있다.

{{< eqlabel wake-phase-kl >}}
$$
\mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [C(\mathbf{x})] = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})) \right] + D_{\mathrm{KL}}(p_{\mathrm{data}}(\mathbf{x}) \| p_{\theta}(\mathbf{x})) + \mathrm{const.}
$$

즉, wake phase는 $\theta$에 대해 두 가지를 동시에 최적화한다. 첫 번째 항은 $q_{\phi}$와 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$ 사이의 평균 KL divergence로, 이를 최소화하면 현재의 $q_{\phi}$에 가장 가까운 posterior를 만드는 $\theta$를 찾을 수 있다. 두 번째 항은 데이터 분포와 모델 분포 사이의 KL divergence로, 이를 최소화하면 데이터를 잘 설명하는 $\theta$를 찾을 수 있다.

최적화 문제를 풀 때는, 먼저 {{< eqref wake-phase >}}에서 $\theta$에 무관한 두 번째 항을 무시해도 된다. 다음으로 $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{z} \mid \mathbf{x}) p_{\theta}(\mathbf{x})$를 대입하면 다음과 같다.

$$
\begin{align*}
\theta
&= \argmin_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [-\log p_{\theta}(\mathbf{x} \mid \mathbf{z}) - \log p_{\theta}(\mathbf{z})]
\end{align*}
$$

이 식은 몬테 카를로 근사로 풀 수 있다. 먼저 $N$개의 샘플 $\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$이 주어졌을 때, 각각의 $n$에 대해 $q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(n)})$에서 $\mathbf{z}^{(n)}$을 샘플링한다. 이제 목적 함수를 다음과 같이 근사할 수 있다.
$$
\begin{align*}
\frac{1}{N} \sum_{n=1}^{N} [-\log p_{\theta}(\mathbf{x}^{(n)} \mid \mathbf{z}^{(n)}) - \log p_{\theta}(\mathbf{z}^{(n)})]
\end{align*}
$$

## Sleep Phase

Sleep phase에서는 $\theta$를 고정하고 $\phi$를 학습시킨다. 식 {{< eqref wake-phase >}}의 목적 함수를 $\phi$에 대해 최적화하는 것도 좋은 접근이다. 하지만 이렇게 하면 목적 함수에 $q_{\phi}$에 대한 기댓값이 들어 있게 되어 $\phi$에 대해 최적화하기 난감하다. 지금까지 우리는 이렇게 기댓값을 취하는 분포에 $\phi$가 들어 있을 때 이를 $\phi$에 대해 미분하기 위해 log-derivative trick을 사용했었다. 뒤에서 살펴볼 VAE에서는 reparametrization trick을 사용한다. 하지만 논문에서 사용한 생성 모델(Helmholtz machine)은 reparametrization trick을 적용할 수 있는 형태가 아니었고, log-derivative trick은 강화 학습에서 이미 알려져 있기는 했지만{{< ref 5 >}} 아직 이 분야와는 연결되지 않은 것으로 보인다.

식 {{< eqref wake-phase-kl >}}에서 $\phi$가 등장하는 항은 첫 번째 KL divergence 뿐이다. 이 식이 최소화되려면 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$에 최대한 가깝게 만들어야 한다는 것을 알 수 있다. $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))$를 최소화하는 것은 어렵지만, 두 분포의 위치를 바꾼 $D_{\mathrm{KL}}(p_{\theta}(\mathbf{z} \mid \mathbf{x}) \| q_{\phi}(\mathbf{z} \mid \mathbf{x}))$를 최소화하는 것은 가능하다. 즉, 다음 최적화 문제를 풀고자 한다.

$$
\phi = \argmin_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \left[ D_{\mathrm{KL}}(p_{\theta}(\mathbf{z} \mid \mathbf{x}) \| q_{\phi}(\mathbf{z} \mid \mathbf{x})) \right]
$$

KL divergence를 전개하면, 아래 식에서 두 번째 등호와 같이 $\phi$와 무관한 항 $\log p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 무시할 수 있다. 하지만 바깥쪽 기댓값이 $p_{\mathrm{data}}(\mathbf{x})$에 대해 취해지고 있으므로, 안쪽의 $p_{\theta}(\mathbf{z} \mid \mathbf{x})$에서 샘플링하기 어렵다. Sleep phase의 아이디어는 $p_{\mathrm{data}}(\mathbf{x})$를 $p_{\theta}$에서 샘플링한 **가상의 데이터** $\tilde{\mathbf{x}} \sim p_{\theta}(\tilde{\mathbf{x}})$로 대체하는 것이다.

$$
\begin{align*}
&\argmin_{\phi} \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}({\mathbf{x}})} \left[ D_{\mathrm{KL}}(p_{\theta}(\mathbf{z} \mid {\mathbf{x}}) \| q_{\phi}(\mathbf{z} \mid \tilde{{x}})) \right] \\
= \, &\argmin_{\phi} \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}({\mathbf{x}})} \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid {\mathbf{x}})} [\log p_{\theta}(\mathbf{z} \mid \mathbf{x}) - \log q_{\phi}(\mathbf{z} \mid {\mathbf{x}})]  \\
= \, &\argmin_{\phi} \mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}({\mathbf{x}})} \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid {\mathbf{x}})} [- \log q_{\phi}(\mathbf{z} \mid {\mathbf{x}})]  \\
\approx \, &\argmin_{\phi} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\theta}(\tilde{\mathbf{x}})} \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z} \mid \tilde{\mathbf{x}})} [- \log q_{\phi}(\mathbf{z} \mid \tilde{\mathbf{x}})]  \\
= \, &\argmin_{\phi} \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z})} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\theta}(\mathbf{x} \mid \mathbf{z})} [- \log q_{\phi}(\mathbf{z} \mid \tilde{\mathbf{x}})]
\end{align*}
$$

위 식 세 번째 등호에서는 $p_{\mathrm{data}}(\mathbf{x})$를 가상의 데이터의 분포 $p_{\theta}(\tilde{\mathbf{x}})$로 대체했다. 네 번째 등호에서는 $p_{\theta}(\tilde{\mathbf{x}}) p_{\theta}(\mathbf{z} \mid \tilde{\mathbf{x}}) = p_{\theta}(\tilde{\mathbf{x}}, \mathbf{z}) = p_{\theta}(\mathbf{z}) p_{\theta}(\tilde{\mathbf{x}} \mid \mathbf{z})$임을 이용해 기댓값의 분포를 계산 가능한 형태로 바꾸었다. 따라서, sleep phase의 목적 함수는 다음과 같다.

{{< eqlabel sleep-phase >}}
$$
\phi = \argmin_{\phi} \mathbb{E}_{\mathbf{z} \sim p_{\theta}(\mathbf{z})} \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\theta}(\mathbf{x} \mid \mathbf{z})} [- \log q_{\phi}(\mathbf{z} \mid \tilde{\mathbf{x}})]
$$

이 식에서는 기댓값을 취하는 분포가 $\theta$이므로, $\phi$에 대해 최적화하기 수월하다.

최적화 문제를 푸는 방법은 다음과 같다. 먼저, $p_{\theta}(\mathbf{z})$에서 $M$개의 샘플 $\mathbf{z}^{(1)}$, $\cdots$, $\mathbf{z}^{(M)}$을 샘플링한다. 다음으로, 각 $m$에 대해 $p_{\theta}(\mathbf{x} \mid \mathbf{z}^{(m)})$에서 $\tilde{\mathbf{x}}^{(m)}$을 샘플링한다. 이제 목적 함수를 다음과 같이 몬테 카를로 근사할 수 있다.

$$
\frac{1}{M} \sum_{m = 1}^{M} -\log q_{\phi}(\mathbf{z}^{(m)} \mid \tilde{\mathbf{x}}^{(m)})
$$


## Wake-Sleep의 분석

Wake-sleep algorithm은 직관적이지만 두 가지 문제가 있다. 먼저, wake phase와 sleep phase의 목적 함수가 다르다는 것이다. 특히, wake phase에서는 $D_{\mathrm{KL}}(q_{\phi} \| p_{\theta})$를 최소화하는 반면 sleep phase에서는 $D_{\mathrm{KL}}(p_{\theta} \| q_{\phi})$를 최소화한다. KL divergence는 비대칭적이므로, 두 phase가 서로 상충하는 방향으로 작용할 수 있다.

또한, sleep phase는 실제 데이터가 아닌 $p_{\theta}$로 생성한 가상의 데이터로 $q_{\phi}$ 를 학습시키므로, $p_{\theta}$가 아직 미숙한 학습 초기에는 엉뚱한 데이터로 학습되는 문제가 있다.

이어서 살펴볼 VAE는 이 두 문제를 모두 해결한다.

# Variational Autoencoder

드디어 VAE를 살펴볼 차례이다{{< ref 1 >}}. VAE의 핵심은 두 가지이다.

1. ELBO를 활용해, $\theta$와 $\phi$가 **동일한 목적 함수**를 공유하도록 한다.
2. **Reparametrization trick**을 이용해 gradient를 계산한다.

핵심 1은 wake-sleep algorithm의 문제점을 해결한다. 핵심 2는 포스트의 맨 위에서 짚었던 높은 분산 문제(문제점 2)를 해결한다. 두 가지 내용을 하나씩 살펴보자.

## VAE의 목적 함수

### Variational Bayes로부터의 유도

VAE의 목적 함수를 유도해 보자. 모든 과정을 한눈에 살펴보기 위해, 번거롭더라도 지금까지 여러 번 반복했던 과정을 다시 한번 반복할 것이다. 먼저, 우리의 목표는 다음과 같은 KL divergence 최소화이다.
$$
\theta = \argmin_{\theta} D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\theta})
$$

KL divergence 최소화는 MLE와 같으므로,
$$
\theta = \argmax_{\theta} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\log p_{\theta}(\mathbf{x})]
$$

이제 편의상 $\log p_{\theta}(\mathbf{x})$만 떼어 놓고 생각하자. 이 값은 marginal log-likelihood라고 부르는데, 다음과 같이 복잡한 적분으로 표현되기 때문에 다루기 어렵다.

$$\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \log \int p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z}) \, d\mathbf{z}$$

한편, 이전 포스트에서 posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$를 통해 marginal likelihood를 다루는 방법을 이야기했다. 그리고 posterior를 다루는 베이지안 추론 방법 중 매개화된 확률 분포 $q_{\phi}$를 이용해 posterior를 근사하는 variational Bayes에 대해 알아보았다. 그리고 이번 포스트에서 하나의 매개변수를 이용해 모든 $\mathbf{x}$에 대한 posterior를 근사하는 amortized inference를 다루었다. Amortized inference를 적용한 variational Bayes의 목적 함수는 다음과 같이 정의된다.

$$
\begin{align*}
\phi &= \argmax_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\mathcal{L}(\phi; \mathbf{x}, \theta)]
\end{align*}
$$

여기에서 $\mathcal{L}$은 ELBO라고 하는데, 다음과 같이 정의된다.
$$
\mathcal{L}(\phi; \mathbf{x}, \theta) = \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
$$

ELBO에는 다음과 같은 성질이 있다. (고정된) $\mathbf{x}$에 대해 ELBO를 최대화하는 것은 $ D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))$를 최소화하는 것과 같다.
{{< eqlabel elbo-1 >}}
$$
\log p_{\theta}(\mathbf{x}) = D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})) + \mathcal{L}(\phi; \mathbf{x}, \theta)
$$
뿐만 아니라, KL divergence는 항상 음이 아닌 실수이므로, 아래 부등식이 성립한다. 따라서 ELBO는 항상 $\log p_{\theta}(\mathbf{x})$의 lower bound이다.
{{< eqlabel elbo-2 >}}
$$
\log p_{\theta}(\mathbf{x}) \ge \mathcal{L}(\phi; \mathbf{x}, \theta)
$$

지금까지는 ELBO를 $\mathcal{L}(\phi; \mathbf{x}, \theta)$로 쓰고 있었는데, $\phi$는 variational Bayes에서 최적화할 매개변수이고 $\mathbf{x}$와 $\theta$는 고정된 값이기 때문이다. 이제 베이지안 추론 문제에서 생성 문제로 다시 돌아오자. $\theta$도 우리가 구하고 싶은 매개변수이므로, 이제 $\mathcal{L}(\phi, \theta; \mathbf{x})$라고 쓰자. VAE는 ELBO를 $\phi$뿐만 아니라 $\theta$에 대해서도 **동시에** 최대화한다.

{{< eqlabel vae-objective >}}
$$
\theta, \phi = \argmax_{\theta, \phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\mathcal{L}(\phi, \theta; \mathbf{x})]
$$

이것이 왜 우리가 원하는 목적 함수인지 확인해 보자. 먼저, 식 {{< eqref elbo-1 >}}에 의해 ELBO를 $\phi$에 대해 최대화하면 $q_{\phi}$가 posterior에 가까워진다. 또한, 식 {{< eqref elbo-2 >}}에 의해 ELBO를 $\theta$에 대해 최대화하면 marginal log-likelihood $\log p_{\theta}(\mathbf{x})$의 lower bound가 올라간다. 물론 $\log p_{\theta}(\mathbf{x})$은 $\mathcal{L}$과 KL divergence만큼 차이가 나기 때문에, $\mathcal{L}$이 증가한다고 해서 $\log p_{\theta}(\mathbf{x})$가 꼭 증가한다는 보장은 없다. $\mathcal{L}$이 증가하는 과정에서 KL divergence가 늘어날 수도 줄어들 수도 있기 때문이다. 하지만 lower bound가 증가하는 것은 항상 보장되므로, $\log p_{\theta}(\mathbf{x})$ 최대화라는 우리의 목표를 근사적으로 달성할 수 있다. 즉, ELBO라는 하나의 목적 함수로 $q_{\phi}$의 posterior 근사와 marginal log-likelihood $\log p_{\theta}(\mathbf{x})$의 최대화라는 **두 가지 목적을 동시에 달성할 수 있다**. 처음 보면 원하는 값이 아니라 그것의 lower bound를 최대화하는 것이 꺼림직하게 느껴질 수 있지만, 식 {{< eqref elbo-1 >}}과 {{< eqref elbo-2 >}}를 살펴보면 매우 깔끔하다.

한 가지 걱정스러운 점이 있을 수 있다. ELBO를 최대화하려고 하는데 $D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x}))$만 계속 줄어들고 정작 우리의 목표인 $\log p_{\theta}(\mathbf{x})$는 그대로일 수도 있지 않나? 이런 일이 일시적으로 일어날 수는 있지만, 영원히 지속될 수는 없다. KL divergence는 항상 $0$ 이상이므로, 줄어들 수 있는 양에 한계가 있기 때문이다. ELBO가 계속 증가하는데 $D_{\mathrm{KL}}$이 이미 $0$에 가까워졌다면, 그 증가분은 $\log p_{\theta}(\mathbf{x})$의 증가에서 올 수밖에 없다.

앞서 살펴보았던 wake-sleep algorithm의 wake phase의 목적 함수인 식 {{< eqref wake-phase >}}의 부호를 바꾸면 식 {{< eqref vae-objective >}}과 똑같다는 사실을 발견할 수 있다. 식 {{< eqref wake-phase >}}는 MDL이라는 다른 접근 방식으로 유도한 식이지만 결과적으로는 ELBO와 동일하다. Wake-sleep algorithm에서는 $\theta$를 최적화하는 wake phase에서만 ELBO를 최적화하고, $\phi$를 최적화하는 sleep phase에서는 다른 목적 함수를 사용했다. 한편, VAE에서는 두 매개변수 모두 ELBO를 사용해 최적화한다. 이렇게 하면 wake-sleep algorithm에서 피하고자 했던, '$\phi$가 들어간 분포에 대한 기댓값을 $\phi$에 대해 최적화'하는 상황에 직면하게 된다. 잠시 후에 이 문제에 대해 더 살펴보자.

### Importance Sampling을 통한 유도

위의 유도 과정은 variational Bayes라는 탄탄한 background에서 ELBO라는 목적 함수를 가져와서 이루어졌다. 그럼에도 불구하고, 위 유도 과정을 처음 보면 ELBO라는 식이 갑자기 뚝 떨어진 것처럼 보인다. 그래서 다른 유도 과정을 소개한다. 이 방법은 marginal likelihood의 intractibility를 직접 해결하기 위해 importance sampling이라는 방법을 도입한다. 위의 유도 과정과 본질적으로는 동일하다.

우리의 목표는 $\log p_{\theta}(\mathbf{x})$를 최대화하는 것이다. Marginal likelihood를 적분으로 쓰면 다음과 같다.

$$
\log p_{\theta}(\mathbf{x}) = \log \int p_{\theta}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
$$

[이전 포스트](../03-variational-autoencoder-1/#intractability-of-marginal-likelihood)에서 이 적분을 직접 계산하기 어려운 이유를 살펴보았다. $p_{\theta}(\mathbf{x}, \mathbf{z}) = p_{\theta}(\mathbf{x} \mid \mathbf{z}) \, p_{\theta}(\mathbf{z})$이므로 적분을 사전 분포 $p_{\theta}(\mathbf{z})$에 대한 기댓값으로 나타낼 수는 있지만, 사전 분포 $p_{\theta}(\mathbf{z})$에서 샘플링한 $\mathbf{z}$가 주어진 $\mathbf{x}$를 잘 설명할 가능성이 매우 낮기 때문에 이 방법은 통하지 않는다. 특정 $\mathbf{x}$에 대해 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 유의미한 값을 가지는 $\mathbf{z}$의 영역은 극히 작아서, 대부분의 샘플에서 $p_{\theta}(\mathbf{x} \mid \mathbf{z}) \approx 0$이 되기 때문이다.

그렇다면, $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 유의미한 값을 가지는 영역에 집중된 분포에서 샘플링할 수 있다면 이 문제를 해결할 수 있지 않을까? 이것이 **importance sampling**의 아이디어이다. 일반적으로, 어떤 함수 $f(\mathbf{z})$의 적분 $\int f(\mathbf{z}) \, d\mathbf{z}$를 임의의 확률 분포 $q(\mathbf{z})$에 대한 기댓값으로 바꿀 수 있다.

$$
\int f(\mathbf{z}) \, d\mathbf{z} = \int \frac{f(\mathbf{z})}{q(\mathbf{z})} q(\mathbf{z}) \, d\mathbf{z} = \mathbb{E}_{\mathbf{z} \sim q(\mathbf{z})} \left[\frac{f(\mathbf{z})}{q(\mathbf{z})}\right]
$$

이렇게 하면 $q$에서 샘플을 뽑아 몬테 카를로 추정으로 적분을 근사할 수 있다. 여기에서 $q$를 **proposal distribution**이라 부른다. Proposal distribution을 잘 선택하면 (즉, $f(\mathbf{z})$가 큰 영역에 집중된 분포를 사용하면) 적은 샘플로도 정확한 근사를 얻을 수 있다.

이 방법을 위의 적분에 적용하자. $\mathbf{x}$에 대한 정보를 반영하는 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 proposal distribution으로 도입하면 다음과 같다.

$$
\log p_{\theta}(\mathbf{x}) = \log \int \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})} q_{\phi}(\mathbf{z} \mid \mathbf{x}) \, d\mathbf{z} = \log \, \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} \left[\frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right]
$$

$\log$는 오목 함수이므로, Jensen's inequality를 적용하면 다음을 얻는다.

$$
\log p_{\theta}(\mathbf{x}) \ge \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} \left[\log \frac{p_{\theta}(\mathbf{x}, \mathbf{z})}{q_{\phi}(\mathbf{z} \mid \mathbf{x})}\right] = \mathcal{L}(\phi, \theta; \mathbf{x})
$$

이렇게 importance sampling과 Jensen's inequality만으로 식 {{< eqref elbo-2 >}}와 동일한 결과를 얻을 수 있다. 이 유도에서 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$의 역할은 다루기 어려운 $\mathbf{z}$ 공간의 적분을 기댓값으로 바꿔 주는 importance sampling의 proposal distribution이다.

그런데 importance sampling은 proposal distribution이 좋아야, 즉 $p_{\theta}(\mathbf{x}, \mathbf{z})$가 큰 영역에 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 집중되어 있어야 효과적이다. $q_{\phi}$가 나쁘면 사전 분포에서 샘플링하는 것과 크게 다르지 않을 수 있다. 하지만 여기에서는 적분값을 직접 근사하는 대신 Jensen's inequality를 적용해 **lower bound**를 구했기 때문에, $q_{\phi}$가 나쁘더라도 부등식 자체는 항상 성립한다. 다만 $q_{\phi}$가 나쁘면 bound가 느슨해져 최적화에 도움이 되지 않으므로, bound를 tight하게 만드는 것이 중요하다.

Jensen's inequality에서 등호가 성립하는 조건은 $\log$ 안의 확률 변수가 상수일 때, 즉 $p_{\theta}(\mathbf{x}, \mathbf{z}) / q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 $\mathbf{z}$에 의존하지 않을 때이다. 이는 $q_{\phi}(\mathbf{z} \mid \mathbf{x}) \propto p_{\theta}(\mathbf{x}, \mathbf{z})$, 즉 $q_{\phi}(\mathbf{z} \mid \mathbf{x}) = p_{\theta}(\mathbf{z} \mid \mathbf{x})$일 때 성립한다. 따라서 $\phi$를 최적화하여 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 true posterior $p_{\theta}(\mathbf{z} \mid \mathbf{x})$에 가깝게 만들수록 bound가 tight해진다. 이는 식 {{< eqref elbo-1 >}}에서 gap이 $D_{\mathrm{KL}}(q_{\phi} \| p_{\theta})$임을 확인한 것과 동일한 결론이다.

### Reconstruction Term과 Regularization Term

이번에는 ELBO를 더 구체적으로 살펴보면서 각 항에 어떤 의미를 부여할 수 있는지 확인해 보자.

$$
\mathcal{L}(\phi, \theta; \mathbf{x}) = \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
$$

$\log p_{\theta}(\mathbf{x}, \mathbf{z}) = \log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \log p_{\theta}(\mathbf{z})$로 쓸 수 있다. 이때 $p_{\theta}(\mathbf{z})$는 $\mathbf{z}$의 밀도함수이기 때문에, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$와 묶어 KL divergence로 나타낼 수 있다.

$$
\begin{align*}
\mathcal{L}(\phi, \theta; \mathbf{x}) &= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p_{\theta}(\mathbf{x} \mid \mathbf{z}) + \log p_{\theta}(\mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]\\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p_{\theta}(\mathbf{x} \mid \mathbf{z})] - \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid\mathbf{x})} \left[\log \frac{q_{\phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z})}\right]\\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})}[\log p_{\theta}(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q_{\phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z}))
\end{align*}
$$

이제 각 항의 의미를 살펴보자. 첫 번째 항의 의미를 풀어서 써 보면 다음과 같다. 
> $\mathbf{x}$가 주어졌을 때, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$로부터 $\mathbf{z}$를 샘플링한 뒤, 이로부터 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 통해 $\mathbf{x}$가 다시 나올 로그 확률밀도의 기댓값

결국 이 항은 $q_{\phi}$가 제공한 $\mathbf{z}$로부터 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$가 원래의 $\mathbf{x}$를 얼마나 잘 복원하는지를 나타낸다. 이 값이 커질수록 복원이 잘 된다는 뜻이므로, 이 항을 **reconstruction term**이라고 부른다.

두 번째 항은 두 확률 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$와 $p_{\theta}(\mathbf{z})$의 KL divergence이다. ELBO를 최대화하면 이 항이 작아지는 방향으로 최적화되므로, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 $\mathbf{z}$의 prior인 $p_{\theta}(\mathbf{z})$에 가까워진다. 그런데 우리는 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 $\mathbf{x}$의 특징을 잘 담은 $\mathbf{z}$를 추출할 수 있기를 원하는데, 이것이 $p_{\theta}(\mathbf{z})$와 비슷하도록 만들어 버리면 $\mathbf{x}$의 정보가 희석되어 버리는 것 아닌가 하는 걱정이 들 수 있다. 실제로 이 항은 각 $\mathbf{x}$에 대해 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 prior 쪽으로 끌어당기는 역할을 하여 reconstruction term과 상충한다. 모델은 이 두 항 사이의 균형을 맞추며, 복원을 위해 $\mathbf{z}$에 $\mathbf{x}$의 정보를 충분히 담으면서도 prior에서 너무 벗어나지 않는 지점을 찾게 된다.

만약 ELBO에 reconstruction term만 있다면, $q_{\phi}(\mathbf{z} \mid \mathbf{x})$는 각 $\mathbf{x}$를 가장 잘 복원하는 $\mathbf{z}$ 하나에 모든 확률을 몰아 주는 방향으로 최적화될 것이다. 이렇게 하면 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$에서 샘플링한 $\mathbf{z}$로는 $p_{\theta}(\mathbf{x} \mid \mathbf{z})$를 통해 의미 있는 $\mathbf{x}$를 생성할 수 있을지 몰라도, $p_{\theta}(\mathbf{z})$에서 샘플링한 $\mathbf{z}$를 사용하면 의미 있는 $\mathbf{x}$를 생성할 수 없게 된다. 두 번째 항은 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 prior $p_{\theta}(\mathbf{z})$에서 너무 벗어나지 않도록 한다. 이렇게 하면 잠재 공간이 prior의 구조를 따르게 되어 생성 모델이 제 기능을 할 수 있게 된다. 이런 의미에서 두 번째 항을 **regularization term**이라고 부른다.

## VAE의 목적 함수 최적화

VAE의 목적 함수를 구체적으로 최적화하는 방법을 알아보자. 먼저 VAE의 목적 함수를 다시 적어 보겠다.
$$
\begin{align*}
J(\theta, \phi) &= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} [\mathcal{L}(\phi, \theta; \mathbf{x})]\\
&= \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})]
\end{align*}
$$

### $\theta$에 대한 최적화

$\theta$에 대한 $J$의 gradient는 다음과 같다. $q_{\phi}(\mathbf{z} \mid \mathbf{x})$는 $\theta$에 의존하지 않으므로 $\nabla_{\theta}$를 기댓값 안으로 넣을 수 있고, ELBO의 두 번째 항인 $\log q_{\phi}(\mathbf{z} \mid \mathbf{x})$도 $\theta$에 무관하므로 사라진다.
$$
\nabla_{\theta} J(\theta, \phi) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\nabla_{\theta} \log p_{\theta}(\mathbf{x}, \mathbf{z})]
$$

$\mathbf{x}^{(1)}$, $\cdots$, $\mathbf{x}^{(N)}$이 $p_{\mathrm{data}}$의 IID 샘플이고, 각 $n$에 대해 $\mathbf{z}^{(n, 1)}$, $\cdots$, $\mathbf{z}^{(n, K)}$가 $q_{\phi}(\mathbf{z} \mid \mathbf{x}^{(n)})$의 IID 샘플일 때, 몬테 카를로 추정량은 다음과 같다.

$$
\nabla_{\theta} J(\theta, \phi) \approx \frac{1}{NK} \sum_{n=1}^{N} \sum_{k=1}^{K} \nabla_{\theta} \log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}^{(n, k)})
$$

여기서 한 샘플 $\mathbf{x}^{(n)}$당 $K$개의 $\mathbf{z}$가 필요해 비효율적이라 생각할 수 있지만, $q_{\phi}$를 식 {{< eqref normal-posterior >}}과 같이 정규 분포로 정의하면 $\boldsymbol{\mu}_{\phi}(\mathbf{x})$와 $\boldsymbol{\Sigma}_{\phi}(\mathbf{x})$를 한 번만 계산하면 $K$개의 $\mathbf{z}$를 모두 샘플링할 수 있으므로 문제가 되지 않는다.

### $\phi$에 대한 최적화

$\phi$에 대한 최적화는 $\theta$와 달리 간단하지 않다. 기댓값을 취하는 분포 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$ 자체가 $\phi$에 의존하므로, $\nabla_{\phi}$를 바로 기댓값 안으로 넣을 수 없기 때문이다.

$$
\nabla_{\phi} J(\theta, \phi) = \nabla_{\phi} \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})] = \, ?
$$

[이전 포스트](../03-variational-autoencoder-1/#eq-high-variance-of-mc)에서 이 상황을 다룬 적이 있다. 그때는 log-derivative trick을 이용해 기댓값 형태의 식을 유도했는데, 여기에서도 다시 한번 시도해 보자. 편의상 $\mathbf{x}$를 고정하고 안쪽 기댓값만 생각하겠다. Log-derivative trick $\nabla_{\phi} q_{\phi} = q_{\phi} \nabla_{\phi} \log q_{\phi}$을 이용하면 다음과 같다. 편의상 중간중간 $q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 그냥 $q_{\phi}$로 썼다.

$$
\begin{align*}
&\nabla_{\phi} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})] \\
&=\nabla_{\phi} \int (\log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi})\,q_{\phi}\,d\mathbf{z} \\
&= \int \nabla_{\phi} q_{\phi} \cdot \log p_{\theta}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} - \int (\nabla_{\phi} q_{\phi} \cdot \log q_{\phi} + q_{\phi} \cdot \nabla_{\phi} \log q_{\phi})\, d\mathbf{z}\\
&= \int q_{\phi} \nabla_{\phi} \log q_{\phi} \cdot \log p_{\theta}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} - \int \left( q_{\phi} \nabla_{\phi} \log q_{\phi} \cdot \log q_{\phi} + \nabla_{\phi} q_{\phi} \right) d\mathbf{z}\\
&= \mathbb{E}_{\mathbf{z} \sim q_{\phi}} \left[ \nabla_{\phi} \log q_{\phi}(\mathbf{z} \mid \mathbf{x}) \cdot \left( \log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x}) \right) \right]
\end{align*}
$$

세 번째 등호에서 log-derivative trick을 사용했고, 네 번째 등호에서 $\int \nabla_{\phi} q_{\phi} \, d\mathbf{z} = 0$을 이용해 마지막 항을 제거했다. 이제 기댓값 형태이므로 몬테 카를로 근사를 적용할 수 있다.

{{< eqlabel grad-phi-with-log-derivative >}}
$$
\nabla_{\phi} J \approx \frac{1}{NK} \sum_{n=1}^{N} \sum_{k=1}^{K} \nabla_{\phi} \log q_{\phi}(\mathbf{z}^{(n, k)} \mid \mathbf{x}^{(n)}) \cdot \left( \log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}^{(n, k)}) - \log q_{\phi}(\mathbf{z}^{(n, k)} \mid \mathbf{x}^{(n)}) \right)
$$

문제는 이 추정량의 분산이 매우 크다는 것이다. 이것이 포스트의 맨 위에서 언급했던 문제점 2이다. VAE에서는 이 문제를 **reparametrization trick**으로 해결한다. 우선 이 해결 방법을 먼저 살펴보고, 이것이 왜 분산을 줄여 주는지는 그 이후에 살펴보자.

## Reparametrization Trick

드디어 꾸준히 언급했던 reparametrization trick을 살펴볼 차례이다. 핵심 아이디어는 간단하다. $\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})$에서 직접 샘플링하는 대신, $\phi$에 의존하지 않는 분포에서 $\boldsymbol{\epsilon}$을 샘플링한 뒤 결정론적 함수 $\mathbf{z} = g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})$를 통해 $\mathbf{z}$를 만드는 것이다.

$q_{\phi}(\mathbf{z} \mid \mathbf{x})$가 식 {{< eqref normal-posterior >}}과 같이 정규 분포라고 가정하자. 그리고 이전 포스트에서 이야기했듯, $\boldsymbol{\Sigma}_{\phi}(\mathbf{x})$가 대각 성분이 양수인 대각행렬이라고 하자. 그러면 다음과 같이 $\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})$를 reparametrize할 수 있다.

$$
\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \mathbf{z} = \boldsymbol{\mu}_{\phi}(\mathbf{x}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}) \odot \boldsymbol{\epsilon}
$$

여기서 $\boldsymbol{\sigma}_{\phi}(\mathbf{x})$는 $\boldsymbol{\Sigma}_{\phi}(\mathbf{x})$의 대각 원소의 제곱근으로 이루어진 벡터이고, $\odot$은 원소별 곱셈이다. 이렇게 정의된 $\mathbf{z}$의 분포가 $\mathcal{N}(\boldsymbol{\mu}_{\phi}(\mathbf{x}), \boldsymbol{\Sigma}_{\phi}(\mathbf{x}))$와 동일하다는 것은 쉽게 확인할 수 있다. 앞으로 이 관계를 $\mathbf{z}_{\phi}(\boldsymbol{\epsilon}) = \boldsymbol{\mu}_{\phi}(\mathbf{x}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}) \odot \boldsymbol{\epsilon}$으로 표기하겠다. 이 표기는 $\mathbf{z}$가 $\phi$에 의존하는 결정론적 함수임을 명시적으로 보여 준다.

이제 $\mathbf{z}$에 대한 기댓값이 주어지면 $\boldsymbol{\epsilon}$에 대한 기댓값으로 바꿀 수 있다. 아래 식에서 $f$는 임의의 함수이다.

$$
\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z} \mid \mathbf{x})} [f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} [f(\mathbf{z}_{\phi}(\boldsymbol{\epsilon}))]
$$

오른쪽 식에서는 기댓값을 취하는 분포 $\mathcal{N}(\mathbf{0}, \mathbf{I})$가 $\phi$에 의존하지 않으므로, $\nabla_{\phi}$를 기댓값 안으로 넣을 수 있다.

$$
\nabla_{\phi} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} [f(\mathbf{z}_{\phi}(\boldsymbol{\epsilon}))] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} [\nabla_{\phi} f(\mathbf{z}_{\phi}(\boldsymbol{\epsilon}))]
$$

이를 ELBO에 적용하면, $\phi$에 대한 gradient는 다음과 같다. 위 식에서 $f(\mathbf{z}) = \log p_{\theta}(\mathbf{x}, \mathbf{z}) - \log q_{\phi}(\mathbf{z} \mid \mathbf{x})$로 놓은 것이다.
$$
\nabla_{\phi} J(\theta, \phi) = \mathbb{E}_{\mathbf{x} \sim p_{\mathrm{data}}(\mathbf{x})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \left[ \nabla_{\phi} \left( \log p_{\theta}(\mathbf{x}, \mathbf{z}_{\phi}(\boldsymbol{\epsilon})) - \log q_{\phi}(\mathbf{z}_{\phi}(\boldsymbol{\epsilon}) \mid \mathbf{x}) \right) \right]
$$
$\nabla_{\phi}$는 $\mathbf{z}_{\phi}(\boldsymbol{\epsilon})$를 통해 간접적으로, 그리고 $\log q_{\phi}$를 통해 직접적으로 $\phi$에 작용한다. 기댓값 안의 식 전체에 gradient가 씌워져 있는데, 이 값은 backpropagation 알고리즘으로 계산할 수 있다. (그래서 더 이상 전개하지 않아도 된다.)

몬테 카를로 추정량은 $\theta$의 경우와 마찬가지이다. $\boldsymbol{\epsilon}^{(n, k)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에서 샘플링한 뒤 $\mathbf{z}^{(n, k)} = \mathbf{z}_{\phi}(\boldsymbol{\epsilon}^{(n, k)}) = \boldsymbol{\mu}_{\phi}(\mathbf{x}^{(n)}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}^{(n)}) \odot \boldsymbol{\epsilon}^{(n, k)}$으로 계산하면 된다.

{{< eqlabel grad-phi-with-reparametrization >}}
$$
\nabla_{\phi} J(\theta, \phi) \approx \frac{1}{NK} \sum_{n=1}^{N} \sum_{k=1}^{K} \nabla_{\phi} \left( \log p_{\theta}(\mathbf{x}^{(n)}, \mathbf{z}_{\phi}(\boldsymbol{\epsilon}^{(n, k)})) - \log q_{\phi}(\mathbf{z}_{\phi}(\boldsymbol{\epsilon}^{(n, k)}) \mid \mathbf{x}^{(n)}) \right)
$$

이 추정량의 분산은 식 {{< eqref grad-phi-with-log-derivative >}}보다 작다.

## 몬테 카를로 추정량의 분산

식 {{< eqref grad-phi-with-reparametrization >}}의 분산이 {{< eqref grad-phi-with-log-derivative >}}보다 작은 이유를 설명하기는 쉽지 않다. VAE 논문에서조차 log-derivative trick으로 구한 추정량의 분산이 크다는 사실만 언급할 뿐, 그 이유를 설명하지는 않는다. 아마 당시에는 이 사실이 직관적이나 실험적으로 확인되었을 뿐 구체적인 분석이 이루어진 것 같지는 않다. 여기서는 두 가지 방법으로 설명하겠다. 첫 번째는 직관적인 설명, 두 번째는 간단한 1차원 정규 분포를 이용한 numerical example이다.

상황을 단순화하자. 두 매개화된 분포 $p_{\theta}(\mathbf{z})$와 $q_{\phi}(\mathbf{z})$가 있다. 이 중 $q_{\phi}(\mathbf{z})$는 표준 정규 분포로 reparametrization이 가능하다. 즉, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$와 결정론적 함수 $g_{\phi}$를 통해 $\mathbf{z} = g_{\phi}(\boldsymbol{\epsilon})$로 나타낼 수 있다. 이때 다음 두 기댓값에 관심을 가지자.
$$
\nabla_{\theta} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})}[\log p_{\theta}(\mathbf{z})], \qquad \nabla_{\phi} \mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})}[\log p_{\theta}(\mathbf{z})]
$$

같은 기댓값을 첫 번째는 $\theta$로, 두 번째는 $\phi$로 미분한 것이다. 이제 두 번째 기댓값을 구하는 방법은 다음 두 가지이다. 첫 번째는 log-derivative trick, 두 번째는 reparametrization trick을 이용한 것이다.

$$
\mathbb{E}_{\mathbf{z} \sim q_{\phi}(\mathbf{z})}[\nabla_{\phi} \log q_{\phi}(\mathbf{z}) \cdot \log p_{\theta}(\mathbf{z})], \qquad  \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)}[\nabla_{\phi} \log p_{\theta}(g_{\phi}(\boldsymbol{\epsilon}))]
$$

이제 세 식의 몬테 카를로 추정량의 분산을 비교하기 위해서는 다음 세 가지 추정량의 분산을 비교해야 한다.

* 추정량 1: $\mathbf{z} \sim q_{\phi}(\mathbf{z})$일 때 $\nabla_\theta \log p_{\theta}(\mathbf{z})$
* 추정량 2: $\mathbf{z} \sim q_{\phi}(\mathbf{z})$일 때 $\nabla_{\phi} \log q_{\phi}(\mathbf{z}) \cdot \log p_{\theta}(\mathbf{z})$
* 추정량 3: $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$일 때 $\nabla_{\phi} \log p_{\theta}(g_{\phi}(\boldsymbol{\epsilon}))$

추정량 1은 $\nabla_{\theta} J(\theta, \phi)$의 몬테 카를로 추정량이고, 추정량 2와 3은 각각 log-derivative trick과 reparametrization trick으로 구한 $\nabla_{\phi} J(\theta, \phi)$의 몬테 카를로 추정량이다. 추정량 1과 3의 분산은 비슷한 수준인 반면, 추정량 2의 분산은 이보다 훨씬 크다는 것을 확인해 보자.

### 직관적인 설명

추정량 1은 $\nabla_\theta \log p_\theta(\mathbf{z})$이다. 추정량 3을 chain rule을 이용해 다시 쓰면 $\nabla_{\phi} \log p_{\theta}(g_{\phi}(\boldsymbol{\epsilon})) = \nabla_\mathbf{z} \log p_\theta(\mathbf{z}) \cdot \nabla_\phi g_\phi(\boldsymbol{\epsilon})$이다. 형태는 다르지만, 둘 다 $\log p_\theta$를 **미분한 값**만 사용한다는 공통점이 있다. 두 식 모두 샘플링된 점 $\mathbf{z}$ 근방에서 $\log p_\theta$가 어떻게 변하는지를 나타내는 국소적 기울기 정보이다.

반면, 추정량 2인 $\nabla_\phi \log q_\phi(\mathbf{z}) \cdot \log p_\theta(\mathbf{z})$에는 $\log p_\theta$의 기울기가 아닌 $\log p_\theta(\mathbf{z})$의 **값 자체**가 곱해져 있다. 이것이 왜 문제가 되는지 살펴보자.

추정량 1이나 3에 포함된 $\log p_\theta(\mathbf{z})$의 gradient는 기본적으로 '어느 방향으로 움직여야 $p_\theta(\mathbf{z})$가 증가하는가'라는 정보를 준다. 하지만 추정량 2에 들어 있는 $\log p_\theta(\mathbf{z})$는 '이 $\mathbf{z}$가 얼마나 좋은가'라는 스칼라 평가만 줄 뿐, 어느 방향으로 개선할 수 있는지는 알려주지 않는다. 물론 score function $\nabla_\phi \log q_\phi(\mathbf{z})$와 곱해져서 간접적으로 gradient 정보가 만들어진다. 하지만 이는 샘플링한 $\mathbf{z}$ 중에서 나쁜 $\mathbf{z}$에서 좋은 $\mathbf{z}$ 방향으로 확률 분포 $q_{\phi}$를 움직이는 역할을 하는데, 여기에서 $p_{\theta}$의 기울기에 대한 정보를 활용하지 못한다.

다른 말로 하면, 추정량 1이나 3은 $p_{\theta}$의 기울기를 활용하는데, 이는 샘플링한 점 $\mathbf{z}$뿐만 아니라 그 주변 국소적인 영역에서의 정보도 활용하는 것이다. 하지만 추정량 2는 $p_{\theta}$의 값만 활용하기 때문에, 샘플링한 점 $\mathbf{z}$들에서 $p_{\theta}$의 크기 차이만 활용할 뿐 그 주변의 정보는 활용하지 못한다. 이렇듯 추정량 2가 샘플을 통해 얻는 정보가 더 적으므로, 비슷한 정확도를 달성하기 위해 더 많은 샘플이 필요하다. 더 많은 샘플이 필요하다는 것은 분산이 크다는 것이다.

정리하면, 추정량 2의 분산이 큰 근본적 이유는 $\log p_\theta$의 값 자체가 들어 있기 때문이다. 추정량 1과 3은 $\log p_\theta$의 gradient만 사용하므로 이 문제를 피할 수 있다.

### Numerical Example

1차원 정규 분포를 가정해 $q_{\phi}(z) = \mathcal{N}(\phi, 1)$, $p_{\theta}(z) = \mathcal{N}(\theta, 1)$로 놓자. Reparametrization은 $g_{\phi}(\epsilon) = \phi + \epsilon$이다 ($\epsilon \sim \mathcal{N}(0, I)$). 두 확률 분포의 로그 밀도 함수는 다음과 같다.

$$
\log q_{\phi}(z) = -\frac{1}{2}\log(2\pi) - \frac{(z - \phi)^{2}}{2}, \quad \log p_{\theta}(z) = -\frac{1}{2}\log(2\pi) - \frac{(z - \theta)^{2}}{2}
$$

**추정량 1: $\theta$에 대한 gradient의 분산.** 
$$
\nabla_{\theta} \log p_{\theta}(z) = z - \theta
$$
이므로 $\text{Var}_{z \sim q_{\phi}}[\nabla_{\theta} \log p_{\theta}(z)] = \text{Var}_{z \sim q_{\phi}}[z] = 1$이다.

**추정량 3: Reparametrization trick의 분산.** $\epsilon \sim \mathcal{N}(0, 1)$, $z = \phi + \epsilon$로 놓으면
$$
\nabla_{\phi} \log p_{\theta}(g_{\phi}(\epsilon)) = \nabla_{\phi} \left[-\frac{(\phi + \epsilon - \theta)^{2}}{2}\right] = -(\phi + \epsilon - \theta)
$$
이므로 $\text{Var}_{\epsilon \sim \mathcal{N}(0,1)}[\theta - \phi - \epsilon] = \text{Var}_{\epsilon \sim \mathcal{N}(0,1)}[\epsilon] = 1$이다. 1번과 동일하다.

**추정량 2: Log-derivative trick의 분산.** Reparametrization은 아니어도 $\epsilon$을 이용해 식을 정리할 수 있다.

$$
\nabla_{\phi} \log q_{\phi}(z) = z - \phi = \epsilon, \quad \log p_{\theta}(z) = -\frac{1}{2}\log(2\pi) - \frac{(\epsilon + \phi - \theta)^{2}}{2}
$$

추정량은 다음과 같다.

$$
\begin{align*}
X &= \nabla_{\phi} \log q_{\phi}(z) \cdot \log p_{\theta}(z) \\
&= \epsilon \cdot \left(-\frac{1}{2}\log(2\pi) - \frac{(\epsilon + \phi - \theta)^{2}}{2}\right)\\
&= \left(C - \frac{(\phi - \theta)^{2}}{2}\right)\epsilon - (\phi - \theta)\epsilon^{2} - \frac{\epsilon^{3}}{2}
\end{align*}
$$

위 식에서 $C = -\frac{1}{2}\log(2\pi)$로 놓았다. $\mathbb{E}_{\epsilon}[\epsilon^{2}] = 1$, $\mathbb{E}_{\epsilon}[\epsilon^{4}] = 3$, $\mathbb{E}_{\epsilon}[\epsilon^{6}] = 15$를 이용해 $X$의 분산을 구해 보자. 편의상 $\phi = \theta$일 때만 구했다.

$$
\begin{align*}
\text{Var}_{\epsilon}[X]
&= \mathbb{E}_{\epsilon}[X^{2}] - (\mathbb{E}_{\epsilon}[X])^{2} \\
&= \mathbb{E}_{\epsilon}\left[\left(C\epsilon - \frac{\epsilon^{3}}{2}\right)^{2}\right] - 0 \\
&= C^{2} \mathbb{E}_{\epsilon}[\epsilon^{2}] - C \, \mathbb{E}_{\epsilon}[\epsilon^{4}] + \frac{1}{4} \mathbb{E}_{\epsilon}[\epsilon^{6}] \\
&= C^{2} - 3C + \frac{15}{4} \\
&\approx 0.84 + 2.76 + 3.75 = 7.35
\end{align*}
$$

표로 정리하면 다음과 같다.

| 추정량 | 분산 ($\theta = \phi$일 때) |
|---|---|
| 추정량 1: $\theta$ gradient | $1$ |
| 추정량 2: Log-derivative trick | $\approx 7.35$ |
| 추정량 3: Reparametrization trick | $1$ |

Reparametrization trick의 분산은 $\theta$ gradient의 분산과 동일하고, log-derivative trick의 분산은 7배 이상 크다. 이 차이는 $\phi \ne \theta$이거나 $\mathbf{z}$가 다차원일 때 더 커질 것이다.

# 정리

이 포스트에서는 이전 포스트에 이어 latent variable model을 살펴보았다. 먼저, 베이지안 추론을 이용해 latent variable model의 posterior를 다루려고 할 때 발생하는 세 가지 문제를 살펴보았다. 우선 MCMC를 버리고 variational Bayes를 채택해 문제점 3을 해결했고, amortized inference를 도입해 문제점 1을 해결했다. 이제 우리의 모델 $p_{\theta}$와 함께 posterior를 근사하는 $q_{\phi}$를 함께 학습시켜야 한다.

그 이후, 서로 다른 목적 함수를 이용해 $\theta$와 $\phi$를 각각 최적화하는 wake-sleep algorithm을 살펴보았다. 다음으로 ELBO라는 하나의 목적 함수를 이용해 두 매개변수를 모두 최적화하는 VAE를 살펴보았다. ELBO를 통해 $\phi$를 최적화하는 과정에서 분산이 지나치게 커져 몬테 카를로 근사를 효율적으로 사용할 수 없는 문제가 발생했다. VAE에서는 이를 reparametrization trick으로 해결했다. 결과적으로, wake-sleep algorithm에서 지적했던 문제들이 모두 해결되었다.

다음 포스트에서는 VAE의 문제점을 살펴보고, 이를 일부 극복하는 Hierarchical VAE (HVAE)에 대해 알아보자.

{{< reflist >}}
{{< refitem 1 >}}
Kingma, Diederik P., and Welling, Max. "[Auto-encoding variational bayes](https://arxiv.org/abs/1312.6114)". *arXiv preprint*, 2013.
{{< /refitem >}}
{{< refitem 2 >}}
Hinton, Geoffrey E., Dayan, Peter, Frey, Brendan J., and Neal, Radford M. "[The "wake-sleep" algorithm for unsupervised neural networks](https://www.science.org/doi/10.1126/science.7761831)". *Science*, 268(5214): 1158–1161, 1995.
{{< /refitem >}}
{{< refitem 3 >}}
Lai, Chieh-Hsin, Song, Yang, Kim, Dongjun, Mitsufuji, Yuki, and Ermon, Stefano. "[The principles of diffusion models](https://arxiv.org/abs/2510.21890)". *arXiv preprint*, 2025.
{{< /refitem >}}
* 이 시리즈의 전반적인 내용을 참고했다.
{{< refitem 4 >}}
Hinton, Geoffrey E., and van Camp, Drew. "[Keeping neural networks simple by minimizing the description length of the weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf)". *Proceedings of the sixth annual conference on Computational learning theory*, pp. 5–13, 1993.
{{< /refitem >}}
{{< refitem 5 >}}
Williams, Ronald J. "[Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696)". *Machine Learning*, 8(3–4): 229–256, 1992.
{{< /refitem >}}
{{< /reflist >}}