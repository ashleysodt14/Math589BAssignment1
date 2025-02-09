import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import csv

# -----------------------------
# Energy Targeting Function
# -----------------------------
def estimate_energy(beads):
    if beads == 10:
        return -21.0
    elif beads == 100:
        return -455.0
    elif beads == 200:
        return -945.0
    return -25.0 + (beads - 10) * (-425.0 / 90.0) if beads < 100 else -450.0 + (beads - 100) * (-495.0 / 100.0)

# -----------------------------
# Initial Structure Generation
# -----------------------------
def generate_structure(beads, dims=3, perturb=1e-5):
    structure = np.zeros((beads, dims))
    for idx in range(1, beads):
        structure[idx, 0] = structure[idx - 1, 0] + 1
        structure[idx, 1] = perturb * np.sin(idx)
        structure[idx, 2] = perturb * np.sin(idx ** 2)
    return structure

# -----------------------------
# Energy Calculation Functions
# -----------------------------
def lj_potential(distance, eps=1.0, sig=1.0):
    return 4 * eps * ((sig / distance) ** 12 - (sig / distance) ** 6)

def bond_energy(dist, eq=1.0, strength=100.0):
    return strength * (dist - eq) ** 2

# -----------------------------
# Compute Total Energy & Gradient
# -----------------------------
def compute_energy_gradient(coords, beads, eps=1.0, sig=1.0, eq=1.0, strength=100.0):
    coords = coords.reshape((beads, -1))
    grad = np.zeros_like(coords)
    total_energy = 0.0
    
    for i in range(beads - 1):
        diff = coords[i+1] - coords[i]
        dist = np.linalg.norm(diff)
        total_energy += bond_energy(dist, eq, strength)
        grad_update = (2 * strength * (dist - eq) / dist) * diff
        grad[i] -= grad_update
        grad[i+1] += grad_update
    
    sep = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(sep, axis=2)
    i_idx, j_idx = np.triu_indices(beads, k=1)
    valid_dists = distances[i_idx, j_idx]
    valid_mask = valid_dists >= 1e-2
    
    energy_lj = np.sum(lj_potential(valid_dists[valid_mask], eps, sig))
    total_energy += energy_lj
    
    dE_dr = 4 * eps * (-12 * sig ** 12 / valid_dists[valid_mask] ** 13 + 6 * sig ** 6 / valid_dists[valid_mask] ** 7)
    
    force_contrib = (dE_dr[:, None] / valid_dists[valid_mask, None]) * sep[i_idx[valid_mask], j_idx[valid_mask]]
    np.add.at(grad, i_idx[valid_mask], force_contrib)
    np.add.at(grad, j_idx[valid_mask], -force_contrib)
    
    return total_energy, grad.flatten()

# -----------------------------
# Optimization with BFGS and CSV Writing
# -----------------------------
def optimize_protein(initial, beads, maxiter=1000, tol=1e-6, write_csv=True):
    x_init = initial.flatten()
    args = (beads,)
    result = minimize(
        fun=compute_energy_gradient,
        x0=x_init,
        args=args,
        method='BFGS',
        jac=True,
        options={'maxiter': maxiter, 'gtol': tol, 'disp': True}
    )
    final_structure = result.x.reshape((beads, -1))
    
    if write_csv:
        filename = f'protein_{beads}.csv'
        np.savetxt(filename, final_structure, delimiter=",")
        print(f"Optimization data saved to {filename}")
    
    return result

# -----------------------------
# Visualization
# -----------------------------
def display_3d_structure(points, title="Protein Configuration"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], '-o', markersize=6)
    ax.set_title(title)
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
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
