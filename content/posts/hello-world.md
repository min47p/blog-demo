---
title: "Hello World: A Tour of Blog Features"
date: 2026-02-02
tags: ["mathematics", "demo"]
math: true
description: "A sample post demonstrating math rendering, callout boxes, toggle sections, references, and code blocks."
---

This is a sample post that showcases the features available in this blog.

## Inline and Block Math

Euler's identity is often cited as the most beautiful equation in mathematics: $e^{i\pi} + 1 = 0$.

The Gaussian integral evaluates to:

$$\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$$

The Cauchy-Schwarz inequality states that for any vectors $\mathbf{u}$ and $\mathbf{v}$ in an inner product space:

$$\left| \langle \mathbf{u}, \mathbf{v} \rangle \right|^2 \leq \langle \mathbf{u}, \mathbf{u} \rangle \cdot \langle \mathbf{v}, \mathbf{v} \rangle$$

## Callout Boxes

{{< callout type="note" >}}
This is a **note** callout. Use it to highlight important information.
{{< /callout >}}

{{< callout type="tip" >}}
This is a **tip** callout. Use it for helpful suggestions.
{{< /callout >}}

{{< callout type="warning" >}}
This is a **warning** callout. Use it to flag potential issues.
{{< /callout >}}

{{< callout type="danger" >}}
This is a **danger** callout. Use it for critical warnings.
{{< /callout >}}

## Toggle Sections

{{< toggle title="Proof that √2 is irrational" >}}
Assume for contradiction that $\sqrt{2} = \frac{p}{q}$ where $p, q \in \mathbb{Z}$ with $\gcd(p,q) = 1$.

Then $2q^2 = p^2$, so $p^2$ is even, which means $p$ is even. Write $p = 2k$.

Then $2q^2 = 4k^2$, so $q^2 = 2k^2$, which means $q$ is also even.

This contradicts $\gcd(p,q) = 1$. Therefore $\sqrt{2}$ is irrational. $\blacksquare$
{{< /toggle >}}

{{< toggle title="Solution to the integral" >}}
Using the substitution $u = x^2$, we can evaluate:

$$\int_0^1 x \, e^{x^2} \, dx = \frac{1}{2}(e - 1)$$
{{< /toggle >}}

## Code Blocks

Here is a Python implementation of the Sieve of Eratosthenes:

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(n + 1) if is_prime[i]]
```

## Tables

| Constant | Symbol | Approximate Value |
|----------|--------|-------------------|
| Pi | $\pi$ | 3.14159 |
| Euler's number | $e$ | 2.71828 |
| Golden ratio | $\varphi$ | 1.61803 |

## References

The Gaussian integral{{< ref 1 >}} is fundamental in probability and statistics. The Cauchy-Schwarz inequality{{< ref 2 >}} is one of the most widely used inequalities in mathematics.

{{< reflist >}}
{{< refitem 1 >}}Strang, G. *Calculus*. Wellesley-Cambridge Press, 1991.{{< /refitem >}}
{{< refitem 2 >}}Steele, J. M. *The Cauchy-Schwarz Master Class*. Cambridge University Press, 2004.{{< /refitem >}}
{{< /reflist >}}
