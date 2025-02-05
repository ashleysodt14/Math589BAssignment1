import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree  # Fast nearest-neighbor search

# Initialize protein positions with randomness
def initialize_protein(n_beads, dimension=3, fudge=1e-2):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  
        positions[i, 1] = fudge * np.sin(i)  
        positions[i, 2] = fudge * np.sin(i*i)
    return positions

# Optimized energy function using spatial partitioning (faster pairwise calculations)
def total_energy_optimized(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions = positions.reshape((n_beads, -1))
    energy = 0.0

    # Bond stretching energy (Harmonic potential)
    for i in range(n_beads - 1):
        r = np.linalg.norm(positions[i+1] - positions[i])
        energy += k_b * (r - b) ** 2  

    # Spatial Partitioning: Use KD-Tree for nearest-neighbor interactions (Fast O(N log N))
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=3.0, output_type='ndarray')  # Only check nearby particles

    # Lennard-Jones potential (Non-bonded interactions)
    for i, j in pairs:
        r = np.linalg.norm(positions[i] - positions[j])
        if r > 1e-2:  
            energy += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    return energy

# Optimization function with fast optimization
def optimize_protein(positions, n_beads, write_csv=False, maxiter=2000, tol=1e-4):
    trajectory = []
    
    # Callback function with correct argument handling
    def callback(x, state=None):  # ✅ Fix: Accept `state` but ignore it
        trajectory.append(x.reshape((n_beads, -1)))
        current_energy = total_energy_optimized(x, n_beads)
        print(f"Iteration {len(trajectory)}, Energy: {current_energy}")

    # Use Trust-Region Optimization (`trust-constr`) for better step control
    result = minimize(
        fun=total_energy_optimized,
        x0=positions.flatten(),
        args=(n_beads,),
        method='trust-constr',  # Faster than L-BFGS-B for large problems
        callback=callback,  # ✅ Fix: Now accepts the correct number of arguments
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )

    # Save results if trajectory is available
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
    reference_energies = {10: -20.9, 100: -455.0, 500: -945.0}

    for n_beads in [10, 100, 500]:
        initial_positions = initialize_protein(n_beads)

        print(f"Initial Energy for {n_beads} beads:", total_energy_optimized(initial_positions.flatten(), n_beads))

        result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)

        optimized_positions = result.x.reshape((n_beads, 3))
        final_energy = total_energy_optimized(optimized_positions.flatten(), n_beads)

        print(f"Optimized Energy for {n_beads} beads: {final_energy}")

        # Check if energy meets reference threshold
        if final_energy < reference_energies[n_beads]:
            print(f"✅ Energy below reference energy {reference_energies[n_beads]}, good!")
        else:
            print(f"❌ Energy {final_energy} is above reference energy {reference_energies[n_beads]}, bad!")
