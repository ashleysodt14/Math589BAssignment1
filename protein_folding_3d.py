import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from matplotlib.animation import FuncAnimation

# -----------------------------
# Initialize Polymer Chain with Random Perturbations
# -----------------------------
def initialize_chain(num_units, dims=3, perturb=0.1):
    coords = np.zeros((num_units, dims))
    for idx in range(1, num_units):
        coords[idx, 0] = coords[idx - 1, 0] + 1  # Keep the backbone structure
        coords[idx, 1:] += perturb * (np.random.rand(dims - 1) - 0.5)  # Random small perturbations
    return coords

# -----------------------------
# Compute Energy Components
# -----------------------------
def lj_potential(dist, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / dist) ** 12 - (sigma / dist) ** 6)

def bond_potential(dist, equilibrium=1.0, strength=100.0):
    return strength * (dist - equilibrium) ** 2

def compute_energy_and_gradient(x, num_units, epsilon=1.0, sigma=1.0, b_eq=1.0, k_bond=100.0):
    coords = x.reshape((num_units, -1))
    dims = coords.shape[1]
    energy = 0.0
    gradients = np.zeros_like(coords)
    
    # Bond stretching energy
    for i in range(num_units - 1):
        delta = coords[i + 1] - coords[i]
        dist = np.linalg.norm(delta)
        if dist == 0:
            continue
        energy += bond_potential(dist, b_eq, k_bond)
        force_mag = 2 * k_bond * (dist - b_eq)
        force_vec = (force_mag / dist) * delta
        gradients[i] -= force_vec
        gradients[i + 1] += force_vec

    # Lennard-Jones interactions
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    upper_indices = np.triu_indices(num_units, k=1)
    dist_vals = distances[upper_indices]
    valid_indices = dist_vals >= 1e-2
    lj_vals = lj_potential(dist_vals[valid_indices], epsilon, sigma)
    energy += np.sum(lj_vals)
    lj_forces = 4 * epsilon * (-12 * sigma**12 / dist_vals[valid_indices]**13 + 6 * sigma**6 / dist_vals[valid_indices]**7)
    delta_vectors = diffs[upper_indices]
    force_contributions = (lj_forces[:, None] / dist_vals[valid_indices, None]) * delta_vectors[valid_indices]
    np.add.at(gradients, upper_indices[0][valid_indices], force_contributions)
    np.add.at(gradients, upper_indices[1][valid_indices], -force_contributions)
    
    return energy, gradients.flatten()

# -----------------------------
# Optimization Routine (BFGS and L-BFGS-B with Simulated Annealing)
# -----------------------------
def optimize_protein(initial_coords, num_units, method="BFGS", maxiter=20000, tol=1e-7, use_basinhopping=False, write_csv=False):
    x0 = initial_coords.flatten()
    args = (num_units,)
    
    # Store trajectory for visualization
    trajectory = []

    def callback(xk):
        trajectory.append(xk.reshape((num_units, -1)))

    if use_basinhopping:
        # Use global optimization (Simulated Annealing-like approach)
        minimizer_kwargs = {
            "method": method,
            "jac": True,
            "callback": callback,
            "options": {'maxiter': maxiter, 'disp': True, 'gtol': tol}
        }
        opt_result = basinhopping(
            compute_energy_and_gradient, x0, minimizer_kwargs=minimizer_kwargs, niter=100
        )
    else:
        # Perform optimization
        opt_result = minimize(
            compute_energy_and_gradient,
            x0,
            args=args,
            method=method,
            jac=True,
            callback=callback,
            options={'maxiter': maxiter, 'disp': True, 'gtol': tol}
        )
    
    if write_csv:
        csv_filepath = f'protein_{num_units}.csv'
        np.savetxt(csv_filepath, opt_result.x.reshape((num_units, 3)), delimiter=",")
        print(f'Data saved to {csv_filepath}')
    
    return opt_result, trajectory

# -----------------------------
# Visualization Functions
# -----------------------------
def visualize_3d(coords, title="Chain Structure"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords = coords.reshape((-1, 3))
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def animate_chain_evolution(trajectory, interval=100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], '-o')

    def update(frame):
        coords = trajectory[frame]
        line.set_data(coords[:, 0], coords[:, 1])
        line.set_3d_properties(coords[:, 2])
        ax.set_title(f"Step {frame + 1}/{len(trajectory)}")
        return line,

    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False)
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    num_units = 100  # Can adjust to 10, 100, or 200 for testing
    init_coords = initialize_chain(num_units)

    # Visualize the initial state
    visualize_3d(init_coords, title="Initial Configuration")

    # Run optimization with improved settings
    result, trajectory = optimize_protein(
        init_coords, num_units, method="L-BFGS-B", maxiter=20000, tol=1e-7, use_basinhopping=True, write_csv=True
    )

    # Visualize optimized state
    optimized_coords = result.x.reshape((num_units, 3))
    visualize_3d(optimized_coords, title="Optimized Configuration")

    # Animate chain evolution
    animate_chain_evolution(trajectory)

    print(result)
