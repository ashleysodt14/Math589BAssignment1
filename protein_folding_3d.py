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
    dims = coords.shape[1]
    energy = 0.0
    gradients = np.zeros_like(coords)
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

def bfgs_optimization(objective_function, initial_x, function_args, num_beads, max_iterations=1000, tolerance=1e-6, step_size=1.0, decay_factor=0.5, Wolfe_c1=1e-4):
    """ 
    Implements the BFGS optimization algorithm for minimizing the objective function.

    Parameters:
    - objective_function: The function to minimize, returning both value and gradient.
    - initial_x: Initial guess for the parameters.
    - function_args: Additional arguments passed to the objective function.
    - num_beads: Number of elements in the polymer chain.
    - max_iterations: Maximum number of iterations.
    - tolerance: Convergence criterion for the gradient norm.
    - step_size: Initial step size for the line search.
    - decay_factor: Reduction factor for step size in backtracking.
    - Wolfe_c1: Wolfe condition parameter for sufficient decrease.

    Returns:
    - optimized_x: The optimized parameter values.
    - history: List of parameter values at each iteration.
    """

    # Initialization
    optimized_x = initial_x.copy()
    dim = len(optimized_x)
    inverse_Hessian = np.eye(dim)  # Approximated Hessian inverse
    history = []

    for iteration in range(max_iterations):
        # Evaluate function value and gradient
        function_value, gradient = objective_function(optimized_x, *function_args)
        gradient_norm = np.linalg.norm(gradient)

        # Check for convergence
        if gradient_norm < tolerance:
            print(f"Converged at iteration {iteration}, gradient norm: {gradient_norm:.8e}")
            break

        # Compute search direction
        search_direction = -inverse_Hessian.dot(gradient)

        # Line search using backtracking
        step_length = step_size
        while True:
            new_x = optimized_x + step_length * search_direction
            new_function_value, _ = objective_function(new_x, *function_args)

            if new_function_value <= function_value + Wolfe_c1 * step_length * np.dot(gradient, search_direction):
                break
            step_length *= decay_factor

            # Avoid infinitesimally small step sizes
            if step_length < 1e-12:
                break

        # Update step and position
        step = step_length * search_direction
        optimized_x += step
        new_function_value, new_gradient = objective_function(optimized_x, *function_args)

        # BFGS Hessian update
        gradient_difference = new_gradient - gradient
        step_dot_gradient = np.dot(gradient_difference, step)

        if step_dot_gradient > 1e-10:
            scaling_factor = 1.0 / step_dot_gradient
            identity_matrix = np.eye(dim)
            term1 = identity_matrix - scaling_factor * np.outer(step, gradient_difference)
            term2 = identity_matrix - scaling_factor * np.outer(gradient_difference, step)
            inverse_Hessian = term1.dot(inverse_Hessian).dot(term2) + scaling_factor * np.outer(step, step)

        # Store trajectory for visualization
        history.append(optimized_x.reshape((num_beads, -1)))

        # Logging
        if (iteration + 1) % 50 == 0:
            print(f"Iteration {iteration + 1}: Function Value = {new_function_value:.6f}, ||Gradient|| = {np.linalg.norm(new_gradient):.2e}")

    return optimized_x, history

# -----------------------------
# Optimization Routine
# -----------------------------
def optimize_protein(initial_coords, num_units, maxiter=10000, tol=1e-6, write_csv=False, ref_energy=None):
    x0 = initial_coords.flatten()
    args = (num_units,)
    
    trajectory = []
    
    def callback(xk):
        trajectory.append(xk.reshape((num_units, -1)))
    
    # Use L-BFGS-B instead of BFGS for better stability in large-scale problems
    opt_result = minimize(
        compute_energy_and_gradient,
        x0,
        args=args,
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={'maxiter': maxiter, 'disp': True, 'gtol': tol}
    )

    # Check convergence
    grad_norm = np.linalg.norm(opt_result.jac)
    energy = opt_result.fun
    
    failed_grad_check = grad_norm > tol
    failed_energy_check = ref_energy is not None and energy > ref_energy

    if failed_grad_check or failed_energy_check:
        print(f"Warning: Optimization did not fully converge. Retrying with increased maxiter ({maxiter * 2}).")
        
        opt_result = minimize(
            compute_energy_and_gradient,
            x0,
            args=args,
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={'maxiter': maxiter * 2, 'disp': True, 'gtol': tol}
        )
        
        grad_norm = np.linalg.norm(opt_result.jac)
        energy = opt_result.fun

    # Final check
    if grad_norm > tol:
        print(f"Final Warning: Gradient norm {grad_norm:.8e} still above tolerance {tol:.8e}.")
    if ref_energy is not None and energy > ref_energy:
        print(f"Final Warning: Energy {energy:.8f} above reference energy {ref_energy:.8f}.")

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
    num_units = 100
    init_coords = initialize_chain(num_units)
    visualize_3d(init_coords, title="Initial Configuration")
    ref_energy_10 = -20.9
    ref_energy_100 = -455.0
    ref_energy_200 = -945.0
    result_10, _ = optimize_protein(init_coords[:10], 10, maxiter=2000, tol=1e-6, write_csv=False, ref_energy=ref_energy_10)
    result_100, _ = optimize_protein(init_coords[:100], 100, maxiter=4000, tol=1e-4, write_csv=False, ref_energy=ref_energy_100)
    result_200, _ = optimize_protein(init_coords, 200, maxiter=10000, tol=0.0005, write_csv=True, ref_energy=ref_energy_200)
    visualize_3d(result_200.x.reshape((200, 3)), title="Optimized Configuration")
