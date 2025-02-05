import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Initialize protein positions with more randomness
def initialize_protein(n_beads, dimension=3, fudge=1e-3):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)  # Increase randomness
        positions[i, 2] = fudge * np.sin(i*i)
    return positions

# Corrected energy function (Lennard-Jones + bond energy)
def total_energy_optimized(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = positions.reshape((n_beads, -1))
    energy = 0.0

    # Bond stretching energy
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += k_b * (r - b) ** 2  # Harmonic bond potential

    # Lennard-Jones potential for non-bonded interactions
    for i in range(n_beads):
        for j in range(i + 1, n_beads):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-2:  # Avoid division by zero
                energy += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    print(f"Current Energy: {energy}")  # Debugging: Print energy values
    return energy

# Optimization function with better tracking
def optimize_protein(positions, n_beads, write_csv=False, maxiter=5000, tol=1e-3):
    trajectory = []

    # Callback for tracking progress
    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        current_energy = total_energy_optimized(x, n_beads)
        print(f"Iteration {len(trajectory)}, Energy: {current_energy}")

    # Run L-BFGS optimization
    result = minimize(
        fun=total_energy_optimized,
        x0=positions.flatten(),
        args=(n_beads,),
        method='L-BFGS-B',
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )

    # Ensure trajectory is saved even if optimization stops early
    if trajectory:
        if write_csv:
            csv_filepath = f'protein{n_beads}.csv'
            print(f'Writing data to file {csv_filepath}')
            np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    else:
        print("Warning: No intermediate trajectories available for saving.")

    return result, trajectory

# Main function
if __name__ == "__main__":
    for n_beads in [10, 100, 500]:
        initial_positions = initialize_protein(n_beads)

        print(f"Initial Energy for {n_beads} beads:", total_energy_optimized(initial_positions.flatten(), n_beads))

        result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)

        optimized_positions = result.x.reshape((n_beads, 3))
        print(f"Optimized Energy for {n_beads} beads:", total_energy_optimized(optimized_positions.flatten(), n_beads))

