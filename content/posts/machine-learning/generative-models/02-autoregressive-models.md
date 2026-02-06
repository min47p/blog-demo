---
title: "Generative Models - 02. Autoregressive Models"
date: 2026-02-05
tags: ["machine-learning", "generative-models"]
draft: true
---

첫 번째로 살펴볼 생성 모델은 autoregressive model이다. [지난 포스트](../01-overview/)에서 이미 autoregressive model에 대해서 간단히 살펴보았는데, 이 내용을 다시 정리하고 모델을 어떻게 학습시키는지까지 알아보자.

## Autoregressive Models

Autoregressive model은 다음과 같은 핵심 아이디어에 기반하고 있다.

{{< callout type="Idea" >}}
복잡한 확률 분포를 단순한 확률 분포들의 조합으로 분해하자. 분해된 각 분포가 충분히 단순하다면, 정규화와 샘플링 문제를 쉽게 해결할 수 있다.
{{< /callout >}}

데이터를 $\mathbf{x} = (x_{1}, \cdots, x_{L})$의 형태로 쓸 수 있다고 하자. 우리의 목표는 $\mathbf{x}$의 분포를 $x_{1}$의 분포, $x_{2}$의 분포, ... $x_{L}$의 분포로 분리하는 것이다. 이 $L$개의 원소들이 서로 독립적이라는 보장은 없기 때문에, 이 과정에서 자연스럽게 조건부 확률이 등장하게 된다.

실제 데이터 $\mathbf{x}$의 확률 분포 $p_{\mathrm{data}}({\mathbf{x}})$는 다음과 같이 $L$개의 조건부 확률 분포들의 곱으로 표현할 수 있다. $x_{1}, \cdots, x_{i - 1}$을 $x_{1:i-1}$로 간단하게 표현했다.

$$p_{\mathrm{data}}(\mathbf{x}) = \prod_{i} p_{\mathrm{data}}(x_{i} \mid x_{1}, \cdots, x_{i - 1}) = \prod_{i} p_{\mathrm{data}}(x_{i} \mid x_{1:i-1})$$

우리가 구하고자 하는 매개화된 확률 분포 $p_{\phi}(\mathbf{x})$도 마찬가지로 쓸 수 있다.

$$p_{\phi}(\mathbf{x}) = \prod_{i} p_{\phi}(x_{i} \mid x_{1}, \cdots, x_{i - 1}) = \prod_{i} p_{\phi}(x_{i} \mid x_{1:i-1})$$

이전 포스트에서, 정규화 조건을 만족하고 샘플링이 가능한 분포를 만들기가 어렵다는 논의를 했다. Autoregressive model은 이 문제를 쉽게 해결한다. 개별적인 원소들의 분포 $p_{\phi}(x_{i} \mid x_{1:i-1})$가 정규화 조건을 만족하고 샘플링이 가능하면 전체 분포 $p_{\phi}(\mathbf{x})$도 마찬가지이기 때문이다. 그럼 이제 $p_{\phi}(x_{i} \mid x_{1:i-1})$를 다루기 쉬운 단순한 분포로 설정하면 된다.

다음과 같은 $L$ 번의 단계를 거쳐 $p_{\phi}(\mathbf{x})$를 샘플링할 수 있다.
- $x_{1} \sim p_{\phi}(x_{1})$을 샘플링한다.
- $x_{2} \sim p_{\phi}(x_{2} \mid x_{1})$을 샘플링한다.
- $x_{3} \sim p_{\phi}(x_{3} \mid x_{1:2})$를 샘플링한다.
- $\cdots$
- $x_{L} \sim p_{\phi}(x_{L} \mid x_{1:L-1})$을 샘플링한다.

$p_{\phi}(\mathbf{x})$가 정규화 조건을 만족하는 이유는 위의 샘플링 과정으로부터 바로 알 수 있다. $x_{1}$, $\cdots$, $x_{L}$을 모두 올바른 확률 분포에서 샘플링했기 때문이다. 이들을 합쳐 만든 $\mathbf{x}$ 또한 당연히 올바른 확률 분포를 따르는 확률 변수가 된다.