import matplotlib.pyplot as plt
import numpy as np

def example_GP_plot() -> None:
    """
    Deterministic visualization of a Gaussian error model
    around an arbitrary smooth function y = f(x).
    No sampling.
    """
    # Domain
    x = np.linspace(0, 10, 300)

    # Arbitrary smooth function (change this freely)
    f = 0.3 * x + 0.8 * np.sin(0.8 * x)

    # Gaussian noise model
    sigma = 0.5
    range_for_error = 10 * sigma
    eps = np.linspace(-range_for_error, range_for_error, 1000)

    # Mesh
    X, E = np.meshgrid(x, eps)
    F = np.tile(f, (eps.size, 1))

    # Gaussian likelihood p(y | f(x))
    pdf = (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
        -(E**2) / (2 * sigma**2)
    )

    # Plot
    plt.figure(figsize=(6, 4))
    pcm = plt.pcolormesh(
        X, F + E, pdf,
        shading="auto",
        cmap="viridis"
    )

    # Mean function
    plt.plot(x, f, color="white", linewidth=2, label="mean f(x)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(f.min()-f.min()*0.1,
             f.max()+f.max()*0.1)
    plt.colorbar(pcm, label="Gaussian likelihood")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/gkerr/lab/docs/presentations/2026-00-00_grad_seminar1/figs/example_GP_plot.png", dpi=300)
    #plt.show()



def main():
    example_GP_plot()




if __name__ == "__main__":
    main()
