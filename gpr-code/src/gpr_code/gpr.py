from __future__ import annotations

import math
from dataclasses import dataclass

import gpytorch
import matplotlib.pyplot as plt
import torch


@dataclass(frozen=True)
class Data:
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    true_f_test: torch.Tensor


def make_toy_data(
    n_train: int = 40,
    n_test: int = 400,
    noise_std: float = 0.10,
    seed: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Data:
    device = device or torch.device("cpu")
    g = torch.Generator(device=device).manual_seed(seed)

    # Two dense clusters with a big gap (no data) in the middle
    n_left = int(0.50 * n_train)
    n_right = n_train - n_left

    # Left cluster: [0.00, 0.35]
    x_left = 0.00 + 0.35 * torch.rand(n_left, generator=g, device=device, dtype=dtype)
    # Right cluster: [0.70, 1.00]
    x_right = 0.70 + 0.30 * torch.rand(n_right, generator=g, device=device, dtype=dtype)

    train_x = torch.cat([x_left, x_right]).sort().values

    true_f_train = torch.sin(train_x * (2.0 * math.pi))
    train_y = true_f_train + noise_std * torch.randn_like(train_x, generator=g)

    test_x = torch.linspace(-0.2, 1.2, n_test, device=device, dtype=dtype)
    true_f_test = torch.sin(test_x * (2.0 * math.pi))

    return Data(train_x=train_x, train_y=train_y, test_x=test_x, true_f_test=true_f_test)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.GaussianLikelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp(
    model: ExactGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    iters: int = 250,
    lr: float = 0.1,
) -> None:
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def predict(
    model: ExactGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))  # predictive distribution over y
        mean = pred.mean
        lower, upper = pred.confidence_region()  # mean ± 2 std
    return mean, lower, upper


def modern_plot(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    mean: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    true_f_test: torch.Tensor | None = None,
) -> None:
    tx = train_x.detach().cpu().numpy()
    ty = train_y.detach().cpu().numpy()
    xx = test_x.detach().cpu().numpy()
    mm = mean.detach().cpu().numpy()
    lo = lower.detach().cpu().numpy()
    up = upper.detach().cpu().numpy()
    ff = true_f_test.detach().cpu().numpy() if true_f_test is not None else None

    # Posterior standard deviation (from 95% band: ±2σ)
    sigma = 0.25 * (up - lo)

    # Identify the biggest gap between consecutive training points
    tx_sorted = tx.copy()
    tx_sorted.sort()
    gaps = tx_sorted[1:] - tx_sorted[:-1]
    i = gaps.argmax()
    gap_lo, gap_hi = tx_sorted[i], tx_sorted[i + 1]

    fig = plt.figure(figsize=(10.2, 6.2), dpi=170)
    gs = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.2), hspace=0.10)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)

    def style(a):
        a.set_facecolor("white")
        a.grid(True, which="major", linewidth=1.0, alpha=0.16)
        a.grid(True, which="minor", linewidth=0.6, alpha=0.09)
        a.minorticks_on()
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_alpha(0.25)

    style(ax)
    style(axs)

    # Shade the sparse gap region on both panels
    for a in (ax, axs):
        a.axvspan(gap_lo, gap_hi, alpha=0.10, linewidth=0, label=None)

    # Top panel: function fit
    ax.fill_between(xx, lo, up, alpha=0.22, linewidth=0, label="95% CI")
    ax.plot(xx, mm, linewidth=2.6, label="Predictive mean")
    if ff is not None:
        ax.plot(xx, ff, linewidth=1.6, alpha=0.35, linestyle="--", label="True function")
    ax.scatter(tx, ty, s=30, alpha=0.95, edgecolor="white", linewidth=0.9, label="Train data", zorder=3)

    ax.set_ylabel("y")

    # Bottom panel: uncertainty (sigma)
    axs.plot(xx, sigma, linewidth=2.2, label="Posterior σ(x)")
    axs.set_xlabel("x")
    axs.set_ylabel("σ")

    # Annotate the gap explicitly
    ax.text(
        0.5 * (gap_lo + gap_hi),
        ax.get_ylim()[0] + 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        "sparse region",
        ha="center",
        va="bottom",
        fontsize=10,
        alpha=0.8,
    )

    # Legends (one for each panel, small)
    leg1 = ax.legend(frameon=True, fancybox=True, framealpha=0.92, loc="upper left")
    leg1.get_frame().set_linewidth(0.0)
    leg2 = axs.legend(frameon=True, fancybox=True, framealpha=0.92, loc="upper left")
    leg2.get_frame().set_linewidth(0.0)

    plt.setp(ax.get_xticklabels(), visible=False)
    fig.tight_layout()
    save_path = "/home/gkerr/lab/docs/presentations/2026-00-00_grad_seminar1/figs/gpr_gptorch.png"
    fig.savefig(save_path, dpi=300)
    print(f"Saved GPR plot to: {save_path}")
    plt.close()
    #plt.show()


def main() -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = make_toy_data(device=device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(data.train_x, data.train_y, likelihood).to(device)

    train_gp(model, likelihood, data.train_x, data.train_y, iters=250, lr=0.1)

    mean, lower, upper = predict(model, likelihood, data.test_x)

    modern_plot(
        train_x=data.train_x,
        train_y=data.train_y,
        test_x=data.test_x,
        mean=mean,
        lower=lower,
        upper=upper,
        true_f_test=data.true_f_test,
    )


if __name__ == "__main__":
    main()
