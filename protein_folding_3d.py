if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot_protein_3d(positions, title="Protein Conformation", ax=None):
        positions = positions.reshape((-1, 3))
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=4)
        ax.set_title(title)
        plt.show()

    # Example usage
    n_beads = 200
    initial_positions = generate_structure(n_beads)

    # Fix: Ensure correct parameter passing
    E_initial, _ = compute_energy_gradient(initial_positions.flatten(), n_beads, eps=1.0, sig=1.0, eq=1.0, strength=100.0)
    print(f"Initial energy: {E_initial:.6f}")

    res = optimize_protein(initial_positions, n_beads, write_csv=False, maxiter=10000, tol=0.5e-3)
    print(f"Optimization done. #iterations={res.nit}, final E={res.fun:.6f}")

    # Plot final result
    plot_protein_3d(res.x.reshape((n_beads, -1)), title="Optimized Conformation")
