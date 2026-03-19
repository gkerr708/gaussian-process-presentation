import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
rng = np.random.default_rng(7)

# ---------------------------
# Synthetic data
# ---------------------------
n = 22
x = np.linspace(0, 10, n)
true_w0 = 1.5
true_w1 = 2.0
sigma_noise = 1.2

y_true = true_w0 + true_w1 * x
y = y_true + rng.normal(0, sigma_noise, size=n)

# Design matrix for y = w0 + w1 x
Phi = np.column_stack([np.ones_like(x), x])

# ---------------------------
# Bayesian linear regression
# Prior: w ~ N(m0, S0)
# Likelihood: y | w ~ N(Phi w, sigma^2 I)
# ---------------------------
alpha = 1.0 / 25.0   # prior precision -> fairly broad prior
beta = 1.0 / sigma_noise**2  # noise precision

m0 = np.zeros(2)
S0 = (1.0 / alpha) * np.eye(2)

# Posterior
SN_inv = np.linalg.inv(S0) + beta * Phi.T @ Phi
SN = np.linalg.inv(SN_inv)
mN = SN @ (np.linalg.inv(S0) @ m0 + beta * Phi.T @ y)

# ---------------------------
# Predictive distribution
# ---------------------------
x_plot = np.linspace(-1, 11, 300)
Phi_plot = np.column_stack([np.ones_like(x_plot), x_plot])

pred_mean = Phi_plot @ mN
pred_var = (1.0 / beta) + np.sum((Phi_plot @ SN) * Phi_plot, axis=1)
pred_std = np.sqrt(pred_var)

# Sample a few posterior lines
w_samples = rng.multivariate_normal(mN, SN, size=6)
y_samples = Phi_plot @ w_samples.T

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(x, y, s=45, label="data")
ax.plot(x_plot, true_w0 + true_w1 * x_plot, linewidth=2, linestyle="--", label="true line")
ax.plot(x_plot, pred_mean, linewidth=2.5, label="posterior mean")

ax.fill_between(
    x_plot,
    pred_mean - 2 * pred_std,
    pred_mean + 2 * pred_std,
    alpha=0.2,
    label=r"$\pm 2\sigma$ predictive band",
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("/home/gkerr/lab/docs/presentations/2026-00-00_grad_seminar1/figs/bayesian_lin_reg.png", dpi=300)
#plt.show()

