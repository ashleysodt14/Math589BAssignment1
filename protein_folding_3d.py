import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree  # Fast nearest-neighbor search
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Helper: Target Energy based on n_beads
# -----------------------------
def get_target_energy(n_beads):
    if n_beads == 10:
        return -21.0
    elif n_beads == 100:
        return -455.0
    elif n_beads == 200:
        return -945.0
    else:
        return -450.0 + (n_beads - 100) * (-495.0 / 100.0)

# -----------------------------
# Initialization
# -----------------------------
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i - 1, 0] + 1
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i * i)
    return positions

# -----------------------------
# Potential Energy Functions
# -----------------------------
def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def bond_potential(r, b=1.0, k_b=100.0):
    return k_b * (r - b)**2

# -----------------------------
# Total Energy and Gradient (Vectorized)
# -----------------------------
def total_energy_with_grad(x, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = x.reshape((n_beads, -1))
    energy = 0.0
    grad = np.zeros_like(positions)

    # Bond Stretching (Harmonic)
    bond_vectors = np.diff(positions, axis=0)
    bond_lengths = np.linalg.norm(bond_vectors, axis=1)
    energy += np.sum(k_b * (bond_lengths - b) ** 2)

    # Spatial Partitioning for Lennard-Jones (Fast)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=3.0, output_type='ndarray')

    # Lennard-Jones potential
    for i, j in pairs:
        r_vec = positions[i] - positions[j]
        r = np.linalg.norm(r_vec)
        if r > 1e-2:
            lj_energy = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
            energy += lj_energy
            dE_dr = 4 * epsilon * (-12 * sigma**12 / r**13 + 6 * sigma**6 / r**7)
            grad_contrib = (dE_dr / r) * r_vec
            grad[i] -= grad_contrib
            grad[j] += grad_contrib

    return energy, grad.flatten()

# -----------------------------
# Custom BFGS with Backtracking
# -----------------------------
def bfgs_optimize(func, x0, args, n_beads, maxiter=1000, tol=1e-6):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)
    trajectory = []

    for k in range(maxiter):
        f, g = func(x, *args)
        if np.linalg.norm(g) < tol:
            print(f"BFGS converged at iteration {k}")
            break

        p = -H @ g
        alpha = 1.0
        while True:
            x_new = x + alpha * p
            f_new, _ = func(x_new, *args)
            if f_new <= f + 1e-4 * alpha * np.dot(g, p):
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        s = alpha * p
        x_new = x + s
        f_new, g_new = func(x_new, *args)
        y = g_new - g
        ys = np.dot(y, s)

        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x = x_new
        trajectory.append(x.reshape((n_beads, -1)))

    return x, trajectory

# -----------------------------
# Protein Optimization
# -----------------------------
def optimize_protein(positions, n_beads, write_csv=False, maxiter=10000, tol=1e-4):
    target_energy = get_target_energy(n_beads)
    x0 = positions.flatten()
    args = (n_beads,)

    # Custom BFGS Optimization
    x_opt, traj = bfgs_optimize(total_energy_with_grad, x0, args, n_beads, maxiter=maxiter, tol=tol)
    final_energy, _ = total_energy_with_grad(x_opt, n_beads)

    # Perturb and Restart if Necessary
    if final_energy > target_energy:
        for _ in range(3):
            x_perturbed = x_opt + np.random.normal(scale=1e-1, size=x_opt.shape)
            x_new, traj_new = bfgs_optimize(total_energy_with_grad, x_perturbed, args, n_beads, maxiter=maxiter//2, tol=tol)
            new_energy, _ = total_energy_with_grad(x_new, n_beads)
            if new_energy < final_energy:
                final_energy, x_opt, traj = new_energy, x_new, traj_new
            if final_energy <= target_energy:
                break

    print(f"Final Energy = {final_energy:.6f} (Target = {target_energy})")

    if write_csv:
        np.savetxt(f'protein{n_beads}.csv', traj[-1], delimiter=",")

    return x_opt, traj

# -----------------------------
# Visualization
# -----------------------------
def plot_protein_3d(positions, title="Protein Structure"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    plt.title(title)
    plt.show()

def animate_optimization(trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], '-o')

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])
        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=False)
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    n_beads = 100
    initial_positions = initialize_protein(n_beads)
    plot_protein_3d(initial_positions, title="Initial Configuration")

    x_opt, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)
    plot_protein_3d(x_opt.reshape((n_beads, 3)), title="Optimized Configuration")
    animate_optimization(trajectory)
