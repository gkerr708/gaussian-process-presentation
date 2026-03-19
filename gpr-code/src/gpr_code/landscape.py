from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def make_landscape(n: int = 220, seed: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    x = np.linspace(-3.2, 3.2, n)
    y = np.linspace(-3.2, 3.2, n)
    X, Y = np.meshgrid(x, y)

    # "Cool" landscape: mixture of smooth bumps + ridges + gentle waves
    Z = (
        1.1 * np.exp(-0.55 * ((X + 1.1) ** 2 + (Y + 0.3) ** 2))
        + 0.9 * np.exp(-0.85 * ((X - 1.2) ** 2 + (Y - 0.9) ** 2))
        - 0.7 * np.exp(-0.75 * ((X + 0.4) ** 2 + (Y - 1.3) ** 2))
        + 0.22 * np.sin(1.8 * X) * np.cos(1.4 * Y)
        + 0.15 * np.cos(2.6 * np.hypot(X, Y))
    )

    # Add very subtle, smooth "terrain" texture (kept tiny so it still looks clean)
    texture = rng.normal(0.0, 1.0, size=Z.shape)
    # cheap smoothing (no scipy): blur by repeated neighbor-averaging
    for _ in range(6):
        texture = (
            texture
            + np.roll(texture, 1, 0) + np.roll(texture, -1, 0)
            + np.roll(texture, 1, 1) + np.roll(texture, -1, 1)
        ) / 5.0
    Z = Z + 0.035 * texture

    return X, Y, Z


def modern_3d_plot(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    fig = plt.figure(figsize=(10.5, 6.2), dpi=170)
    ax = fig.add_subplot(111, projection="3d")

    # Surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap="magma",
        rstride=1, cstride=1,
        linewidth=0,
        antialiased=False,   # <- key
        shade=True,
    )

    # Camera / framing
    ax.view_init(elev=40, azim=-55)
    ax.set_box_aspect((1, 1, 0.45))

    # Clean, modern look: kill pane fills + soften axis lines
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.04))  # barely-there backdrop
        axis.pane.set_edgecolor((0, 0, 0, 0.15))
    ax.xaxis.line.set_alpha(0.20)
    ax.yaxis.line.set_alpha(0.20)
    ax.zaxis.line.set_alpha(0.20)

    # Minimal ticks
    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
    ax.set_zticks([])

    ax.set_xlabel("x", labelpad=10)
    ax.set_ylabel("y", labelpad=10)

    # Colorbar, minimal
    cbar = fig.colorbar(surf, ax=ax, shrink=0.68, pad=0.06)
    cbar.outline.set_alpha(0.0)
    cbar.set_label("z", rotation=0, labelpad=10)

    fig.tight_layout()
    savepath = "/home/gkerr/lab/docs/presentations/2026-00-00_grad_seminar1/figs/landscape.png"
    fig.savefig(savepath, dpi=300, bbox_inches="tight", pad_inches=0.02)
    #plt.show()





def main() -> None:
    X, Y, Z = make_landscape()
    modern_3d_plot(X, Y, Z)


if __name__ == "__main__":
    main()
