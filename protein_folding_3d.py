import numpy as np
import time

# Helper functions for energy and gradient computation
def bond_potential(r, b=1.0, k_b=100.0):
    return k_b * (r - b)**2

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = positions.reshape((n_beads, -1))
    energy = 0.0
    # Bond energy
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += bond_potential(r, b, k_b)
    # Lennard-Jones potential for non-bonded interactions
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-2:
                energy += lennard_jones_potential(r, epsilon, sigma)
    return energy

def compute_gradient(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    gradient = np.zeros_like(positions)
    positions = positions.reshape((n_beads, -1))
    # Bond energy gradient
    for i in range(n_beads - 1):
        r_vec = positions[i+1] - positions[i]
        r = np.linalg.norm(r_vec)
        grad = 2 * k_b * (r - b) * r_vec / r
        gradient[i] += grad
        gradient[i+1] -= grad
    # Lennard-Jones energy gradient
    for i in range(n_beads):
        for j in range(i+1, n_beads):
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r > 1e-2:
                grad = 24 * epsilon * (2 * (sigma / r)**12 - (sigma / r)**6) * r_vec / r**2
                gradient[i] += grad
                gradient[j] -= grad
    return gradient.flatten()

def bfgs_with_backtracking(positions, n_beads, maxiter=1000, tol=1e-6, perturb_threshold=100, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    """
    BFGS with Backtracking Line Search and Perturbation if the energy is too high.
    """
    # Initial setup
    positions = positions.flatten()
    trajectory = []
    gradient = compute_gradient(positions, n_beads, epsilon, sigma, b, k_b)
    H_inv = np.eye(len(positions))  # Initial inverse Hessian
    start_time = time.time()

    # Begin BFGS iterations
    for iteration in range(maxiter):
        energy = total_energy(positions, n_beads, epsilon, sigma, b, k_b)
        trajectory.append(positions.reshape(n_beads, -1))
        grad = compute_gradient(positions, n_beads, epsilon, sigma, b, k_b)

        # Check if the total energy is above the threshold, and perturb the positions
        if energy > perturb_threshold:
            print(f"Energy is high, perturbing positions: {energy}")
            perturbation = np.random.normal(scale=1e-2, size=positions.shape)
            positions += perturbation
            continue  # Skip the regular update if perturbing

        # Convergence check
        if np.linalg.norm(grad) < tol:
            print(f"Converged in {iteration} iterations")
            break
        
        # Compute search direction (using the inverse Hessian)
        search_direction = -np.dot(H_inv, grad)
        
        # Backtracking line search
        step_size = 1.0
        while total_energy(positions + step_size * search_direction, n_beads, epsilon, sigma, b, k_b) > energy + 1e-4 * step_size * np.dot(grad, search_direction):
            step_size *= 0.5  # Reduce the step size by half

        # Update the positions with the chosen step size
        new_positions = positions + step_size * search_direction

        # Update the inverse Hessian using the BFGS formula
        delta_x = new_positions - positions
        delta_grad = compute_gradient(new_positions, n_beads, epsilon, sigma, b, k_b) - grad
        rho = 1.0 / np.dot(delta_grad, delta_x)
        H_inv = np.dot(np.eye(len(positions)) - rho * np.outer(delta_x, delta_grad), H_inv)
        H_inv = np.dot(H_inv, np.eye(len(positions)) - rho * np.outer(delta_grad, delta_x))
        H_inv += rho * np.outer(delta_x, delta_x)
        
        positions = new_positions

        # Time check for large n_beads
        elapsed_time = time.time() - start_time
        if elapsed_time > 600:  # Timeout if optimization exceeds 600 seconds
            print("Optimization exceeded time limit (600 seconds).")
            break

    return positions, energy, trajectory

# Example: running the optimization with a small number of beads
if __name__ == "__main__":
    n_beads = 10
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)
    optimized_positions, final_energy, trajectory = bfgs_with_backtracking(initial_positions, n_beads)
    
    print(f"Final Optimized Energy: {final_energy}")
    print("Optimized Positions:", optimized_positions)
