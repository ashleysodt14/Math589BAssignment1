import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ------------------------------
# Helper: Target Energy Function
# ------------------------------
def calculate_target_energy(n_beads):
    """Estimate the target energy based on the number of beads."""
    if n_beads == 10:
        return -20.9
    elif n_beads == 100:
        return -455.0
    elif n_beads == 200:
        return -945.0
    else:
        if n_beads < 100:
            return -20.0 + (n_beads - 10) * (-430.0 / 90.0)
        else:
            return -450.0 + (n_beads - 100) * (-495.0 / 100.0)

# ------------------------------
# Initialization: Position of the Protein
# ------------------------------
def generate_initial_positions(n_beads, dimension=3, fudge=1e-5):
    """Generate initial positions for the beads of the protein."""
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.cos(i * i)
    return positions

# ------------------------------
# Potential Energy Calculations
# ------------------------------
def compute_lennard_jones(r, epsilon=1.0, sigma=1.0):
    """Compute the Lennard-Jones potential between two particles."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def compute_bond_potential(r, b=1.0, k_b=100.0):
    """Compute the bond potential between two adjacent beads."""
    return k_b * (r - b)**2

# ------------------------------
# Total Energy and Gradient Calculation
# ------------------------------
def energy_and_gradient(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """Calculate total energy and its gradient for the protein structure."""
    positions = positions.reshape((n_beads, -1))
    gradient = np.zeros_like(positions)
    energy = 0.0

    # Bond energy contribution
    for i in range(n_beads - 1):
        r_vec = positions[i + 1] - positions[i]
        r = np.linalg.norm(r_vec)
        energy += compute_bond_potential(r, b, k_b)
        grad = 2 * k_b * (r - b) * r_vec / r
        gradient[i] -= grad
        gradient[i + 1] += grad

    # Lennard-Jones potential energy contribution
    diff = positions[:, None, :] - positions[None, :, :]
    r_mat = np.linalg.norm(diff, axis=2)
    idx_i, idx_j = np.triu_indices(n_beads, k=1)
    r_ij = r_mat[idx_i, idx_j]
    valid = r_ij >= 1e-2
    r_valid = r_ij[valid]
    LJ_energy = 4 * epsilon * ((sigma / r_valid)**12 - (sigma / r_valid)**6)
    energy += np.sum(LJ_energy)

    # Gradient for Lennard-Jones
    dE_dr = 4 * epsilon * (-12 * sigma**12 / r_valid**13 + 6 * sigma**6 / r_valid**7)
    diff_ij = diff[idx_i, idx_j]
    contrib = (dE_dr[:, None] / r_valid[:, None]) * diff_ij[valid]
    valid_i = idx_i[valid]
    valid_j = idx_j[valid]
    np.add.at(gradient, valid_i, contrib)
    np.add.at(gradient, valid_j, -contrib)

    return energy, gradient.flatten()

# ------------------------------
# Gradient Descent with Backtracking Line Search
# ------------------------------
def gradient_descent_backtracking(func, x0, args, maxiter=1000, tol=1e-6, alpha0=1.0, beta=0.5, c=1e-4):
    """Gradient descent with backtracking line search."""
    x = x0.copy()
    trajectory = []
    for k in range(maxiter):
        f, g = func(x, *args)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            print(f"Converged at iteration {k} with gradient norm {g_norm:.8e}")
            break
        p = -g  # Gradient descent step
        alpha = alpha0
        while True:
            x_new = x + alpha * p
            f_new, _ = func(x_new, *args)
            if f_new <= f + c * alpha * np.dot(g, p):
                break
            alpha *= beta
        x = x_new
        trajectory.append(x.reshape(args[0], -1))
        if (k + 1) % 50 == 0:
            print(f"Iteration {k+1}: f = {f_new:.6f}, ||g|| = {np.linalg.norm(g):.2e}")
    return x, trajectory

# ------------------------------
# Optimization: BFGS with Backtracking and Perturbations
# ------------------------------
def optimize_protein(positions, n_beads, maxiter=10000, tol=1e-4, target_energy=None, write_csv=False):
    """Optimize the protein configuration using gradient descent with backtracking and perturbations."""
    if target_energy is None:
        target_energy = calculate_target_energy(n_beads)

    x0 = positions.flatten()
    args = (n_beads,)
    
    # Run the gradient descent with backtracking
    x_opt, traj = gradient_descent_backtracking(energy_and_gradient, x0, args, maxiter=maxiter, tol=tol)

    # Evaluate the energy at the final optimized configuration
    final_energy, _ = energy_and_gradient(x_opt, n_beads)
    print(f"Initial energy: {final_energy:.6f}")

    best_energy = final_energy
    best_x = x_opt.copy()

    # Perturbations if energy is too high
    if best_energy > target_energy:
        print(f"Energy exceeds target, performing perturbation...")
        n_perturb = 3
        for i in range(n_perturb):
            print(f"Perturbation {i+1}...")
            perturbation = np.random.normal(scale=1e-2, size=best_x.shape)
            x_perturbed = best_x + perturbation
            x_new, traj_new = gradient_descent_backtracking(energy_and_gradient, x_perturbed, args, maxiter=maxiter // 2, tol=tol)
            f_new, _ = energy_and_gradient(x_new, n_beads)
            if f_new < best_energy:
                best_energy = f_new
                best_x = x_new.copy()
                best_traj = traj_new.copy()

    print(f"Final energy = {best_energy:.6f} (target = {target_energy})")

    # Visualize the optimized structure and animation
    if write_csv:
        csv_filepath = f'optimized_protein_{n_beads}.csv'
        np.savetxt(csv_filepath, best_traj[-1], delimiter=",")
    
    return best_x, best_traj

# ------------------------------
# 3D Visualization of Protein
# ------------------------------
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
    """
    Plot the 3D positions of the protein beads.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# ------------------------------
# Animation of Protein Optimization Process
# ------------------------------
def animate_optimization(trajectory, interval=100):
    """
    Animate the optimization process in 3D.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line, = ax.plot([], [], [], '-o', markersize=6)

    def update(frame):
        positions = trajectory[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        line.set_3d_properties(positions[:, 2])

        # Autoscale the axes
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        ax.set_xlim(x_min - 1, x_max + 1)
        ax.set_ylim(y_min - 1, y_max + 1)
        ax.set_zlim(z_min - 1, z_max + 1)

        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    plt.show()

# ------------------------------
# Main Function to Run the Optimization
# ------------------------------
if __name__ == "__main__":
    # Initialize and set up parameters
    n_beads = 100  # You can change this to 10, 100, or 200 beads based on your test case
    dimension = 3  # 3D protein conformation
    initial_positions = generate_initial_positions(n_beads, dimension)

    # Print initial energy
    initial_energy, _ = energy_and_gradient(initial_positions.flatten(), n_beads)
    print("Initial Energy:", initial_energy)

    # Visualize the initial configuration
    plot_protein_3d(initial_positions, title="Initial Configuration")

    # Run optimization
    result_positions, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True, maxiter=10000, tol=1e-4)

    # Visualize the optimized configuration
    optimized_positions = result_positions.reshape((n_beads, dimension))
    optimized_energy, _ = energy_and_gradient(optimized_positions.flatten(), n_beads)
    print("Optimized Energy:", optimized_energy)
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process
    animate_optimization(trajectory)
