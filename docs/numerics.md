# Numerical Details

## 1. Floating-Point Background

Floating-point formats have finite precision. Values are represented with a fixed number of significant binary digits and a limited exponent range, so each exact real value is rounded to a nearby representable value.

Spacing depends on magnitude. Near `1`, the distance between adjacent representable values is much larger than the smallest representable positive value near zero. Therefore, a small positive number may still be representable by itself while disappearing when added to `1`. Conversely, a tiny gradient can remain representable even after the corresponding probability rounds to one.

Algebraically equivalent formulas can also have different floating-point behavior. In particular, subtracting two nearly equal rounded values can discard their shared leading digits. This is cancellation. It differs from overflow, where a result exceeds the largest representable magnitude, and underflow, where a result is too small to represent (or is rounded to zero).

## 2. Softmax Collapse

For logits \(z\), a hard target \(t\), and probabilities \(q=\operatorname{softmax}(z)\), the usual cross-entropy is

\[
L=-\log q_t.
\]

Its exact gradient is \(g_k=q_k-\mathbf{1}[k=t]\). Define the target margin against the other logits:

\[
m=z_t-\log\sum_{j\ne t}e^{z_j}.
\]

The target probability is \(q_t=1/(1+e^{-m})\), so the target gradient can be written in two equivalent forms:

\[
g_t=q_t-1=-\frac{e^{-m}}{1+e^{-m}}.
\]

The first form subtracts two nearly equal numbers when \(q_t\) is close to one. In common precisions, `1 + exp(-m)` rounds to exactly `1` beyond an approximate margin of 5.5 in bfloat16, 16.6 in float32, or 36.7 in float64. Then \(q_t\) rounds to one and \(q_t-1\) becomes zero. The non-target gradients

\[
g_j=\frac{e^{z_j-z_t}}{1+e^{-m}},\qquad j\ne t,
\]

can remain nonzero because their small exponential numerators are not subtracted from one. The resulting gradient no longer sums to zero.

The fixed formulation computes the margin without placing the target's unit mass inside the LogSumExp:

\[
m=-\operatorname{LSE}_{j\ne t}(z_j-z_t),
\qquad
L=\operatorname{softplus}(-m).
\]

The small quantity stays inside the stable Softplus evaluation. The target logit appears in the loss through the differences \(z_j-z_t\) for \(j\ne t\), so ordinary autograd computes its gradient as the negative sum of the non-target gradients. This preserves the zero-sum relationship without forming `1 - p_target` or adding another gradient projection.

For log-softmax, let \(i_{\max}=\arg\max_i z_i\). Shift all logits by the maximum, explicitly set the selected maximum entry to constant zero, and exclude it from the LogSumExp:

\[
\tilde z_k=\begin{cases}0 & k=i_{\max},\\ z_k-z_{i_{\max}} & k\ne i_{\max},\end{cases}
\qquad
m=-\operatorname{LSE}_{j\ne i_{\max}}\tilde z_j,
\]

\[
\log\operatorname{softmax}(z)_k=\tilde z_k-\operatorname{softplus}(-m).
\]

Replacing the maximum with a constant avoids relying on the differentiable cancellation \(z_{i_{\max}}-z_{i_{\max}}\). `fixed_softmax` exponentiates these log-probabilities.

## 3. Softmax Cancellation

For teacher logits \(z_T\) and student logits \(z_S\), let

\[
p=\operatorname{softmax}(z_T),\qquad q=\operatorname{softmax}(z_S).
\]

The soft-target cross-entropy and its student gradient are

\[
L=-\sum_k p_k\log q_k,
\qquad
\nabla_{z_S}L=q-p.
\]

When the two distributions are close, separately rounding \(p\) and \(q\) before subtracting them loses precision. Even with identical logits, different forward and backward operation sequences can leave a spurious residual.

Instead, compute the difference from logit differences. Let

\[
\delta_k=z_{S,k}-z_{T,k},
\qquad
d_k=\delta_k-\sum_i p_i\delta_i.
\]

The centering removes any shared offset. Since \(\sum_i p_i d_i=0\),

\[
\frac{q_k}{p_k}
=\frac{e^{d_k}}{\sum_i p_i e^{d_i}}
=\exp\!\left(d_k-\log\sum_i p_i e^{d_i}\right).
\]

Use \(e^x=1+\operatorname{expm1}(x)\) and \(\sum_i p_i=1\) to express the log-ratio as

\[
s_k=\log(q_k/p_k)
=d_k-\operatorname{log1p}\!\left(\sum_i p_i\operatorname{expm1}(d_i)\right).
\]

The probability difference is then evaluated as

\[
u_k=p_k\operatorname{expm1}(s_k).
\]

This keeps a small logit difference visible instead of subtracting close rounded probabilities. When \(|s_k|\ge1\), the implementation falls back to \(u_k=q_k-p_k\). In this range the subtraction is no longer between nearly equal values; the fallback also avoids the indeterminate product `0 * inf` if \(p_k\) rounded to zero while `expm1(s_k)` overflowed.

Finite-precision operations can leave a small residual in \(\sum_k u_k\). The implementation applies the paper's correction

\[
g_k=u_k-q_k\sum_i u_i.
\]

Because \(q\) sums to one, this makes the gradient sum to zero in exact arithmetic and reduces residual drift in floating point. At identical student and teacher logits, \(\delta=d=s=u=0\), so the corrected student gradient is exactly zero in the supported computation.

## 4. Stop-Gradient Construction

The forward value must remain the ordinary soft-target cross-entropy, while the student backward value uses \(g\). The implementation forms the forward term with detached student log-probabilities:

\[
L_{\mathrm{forward}}=-\sum_k p_k\,\operatorname{stopgrad}(\log q_k).
\]

It then adds a zero-valued gradient term:

\[
\sum_k \operatorname{stopgrad}(g_k)
\left(z_{S,k}-\operatorname{stopgrad}(z_{S,k})\right).
\]

The added term is zero in the forward pass. Its derivative with respect to student logits is \(g\), and detaching the forward log-probabilities prevents the ordinary student gradient from being added a second time. Teacher probabilities are not detached in the forward loss, so teacher logits that require gradients receive the corresponding forward-objective gradient. Pass detached teacher logits when the teacher is frozen.

The correction is detached and designed for first-order differentiation. This construction does not provide the higher-order derivatives of the exact soft-target objective.

## 5. Precision and Limitations

- The implementation does not silently promote all inputs to float64. `float32`, `float64`, and supported low-precision inputs use ordinary PyTorch operations in their input dtype; device kernels may have their own implementation details.
- bfloat16 has fewer significand bits than float32. The rewrite avoids specific fragile arithmetic, but does not guarantee greater accuracy than every fused or internally promoted PyTorch kernel.
- Tiny gradients eventually underflow. The fix removes the rounding cliff caused by adding a tiny mass to one; it cannot represent values below the dtype's range.
- The code is a reference implementation built from general PyTorch operations. It may use more temporary tensors and run slower than fused kernels.
- Inputs are expected to be finite and the class dimension is the final dimension. Masked negative-infinity logits and special-value behavior are unsupported.
- CUDA and CPU kernels can differ slightly. `torch.compile` and specialized backends have not been separately validated.
- The soft-target gradient substitution supports first-order differentiation only. The repository does not include custom CUDA/Triton kernels or distributed-training support.
