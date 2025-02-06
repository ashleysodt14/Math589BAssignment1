import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation

# -----------------------------
# Initialize Polymer Chain
# -----------------------------
def initialize_chain(num_units, dims=3, perturb=1e-5):
    coords = np.zeros((num_units, dims))
    for idx in range(1, num_units):
        coords[idx, 0] = coords[idx - 1, 0] + 1
        coords[idx, 1] = perturb * np.sin(idx)
        coords[idx, 2] = perturb * np.sin(idx**2)
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
    energy = 0.0
    gradients = np.zeros_like(coords)
    
    # Compute bond energy and gradient
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
    
    # Compute Lennard-Jones energy and gradient
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
# BFGS with Backtracking
# -----------------------------
def bfgs_optimize(func, x0, args, n_beads, maxiter=1000, tol=1e-6, alpha0=1.0, beta=0.5, c=1e-4):
    x = x0.copy()
    n = len(x)
    H = np.eye(n)
    trajectory = []
    for k in range(maxiter):
        f, g = func(x, *args)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            print(f"BFGS converged at iteration {k} with gradient norm {g_norm:.8e}")
            break
        p = -H.dot(g)
        alpha = alpha0
        while True:
            x_new = x + alpha * p
            f_new, _ = func(x_new, *args)
            if f_new <= f + c * alpha * np.dot(g, p):
                break
            alpha *= beta
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
            H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
        trajectory.append(x.reshape((n_beads, -1)))
        if (k+1) % 50 == 0:
            print(f"Iteration {k+1}: f = {f_new:.6f}, ||g|| = {np.linalg.norm(g_new):.2e}")
    return x, trajectory

# -----------------------------
# Optimization Routine
# -----------------------------
def optimize_protein(initial_coords, num_units, maxiter=10000, tol=1e-4, write_csv=False):
    x0 = initial_coords.flatten()
    args = (num_units,)
    trajectory = []
    
    def callback(xk):
        trajectory.append(xk.reshape((num_units, -1)))

    opt_result = minimize(
        compute_energy_and_gradient,
        x0,
        args=args,
        method='BFGS',
        jac=True,
        callback=callback,
        options={'maxiter': maxiter, 'disp': True}
    )
    
    if write_csv:
        csv_filepath = f'protein_{num_units}.csv'
        np.savetxt(csv_filepath, opt_result.x.reshape((num_units, 3)), delimiter=",")
        print(f'Data saved to {csv_filepath}')
    
    return opt_result, trajectory


def verify_gradient(func, x, num_units, epsilon=1.0, sigma=1.0, b_eq=1.0, k_bond=100.0, h=1e-5):
    """
    Verify gradients using the central difference formula.
    
    Parameters:
    - func: Function to compute energy and analytic gradient
    - x: Current flattened positions
    - num_units: Number of units in the chain
    - epsilon, sigma, b_eq, k_bond: Potential parameters
    - h: Step size for finite difference
    """
    analytic_energy, analytic_grad = func(x, num_units, epsilon, sigma, b_eq, k_bond)
    finite_diff_grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += h
        x_backward[i] -= h
        
        f_forward, _ = func(x_forward, num_units, epsilon, sigma, b_eq, k_bond)
        f_backward, _ = func(x_backward, num_units, epsilon, sigma, b_eq, k_bond)
        
        finite_diff_grad[i] = (f_forward - f_backward) / (2 * h)
    
    # Compute the error
    diff = np.linalg.norm(analytic_grad - finite_diff_grad)
    print(f"Gradient verification: ||analytic_grad - finite_diff_grad|| = {diff:.8e}")
    
    return diff


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
    num_units = 100
    init_coords = initialize_chain(num_units)
    visualize_3d(init_coords, title="Initial Configuration")
    
    # Flatten the initial positions
    x0 = init_coords.flatten()
    
    # Verify the gradient
    print("Verifying gradients...")
    grad_diff = verify_gradient(
        compute_energy_and_gradient,
        x0,
        num_units,
        epsilon=1.0,
        sigma=1.0,
        b_eq=1.0,
        k_bond=100.0,
        h=1e-5
    )
    
    if grad_diff < 1e-6:
        print("Gradient verification passed!")
    else:
        print("Warning: Gradient verification failed.")
    
    # Perform optimization using BFGS with backtracking
    print("Optimizing using custom BFGS with backtracking...")
    x_opt, traj = bfgs_optimize(compute_energy_and_gradient, x0, (num_units,), num_units, maxiter=1000)
    
    # Visualize results
    optimized_coords = x_opt.reshape((num_units, 3))
    visualize_3d(optimized_coords, title="Optimized Configuration")
    print("Optimization complete.")
