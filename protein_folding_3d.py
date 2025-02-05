import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import ctypes

# Cython or ctypes integration for energy and distance functions can be placed here
# For now, assuming we have a function `calculate_pairwise_distances` in C to speed up distance calculations

# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge = 1e-5):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i*i)
    return positions

# C-based optimized total energy function (to be implemented using ctypes or cython)
# Assume the existence of a compiled C function to calculate pairwise distances and potential
def total_energy_optimized(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    # This would be a call to the C function to compute the energy efficiently
    # e.g., energy = calculate_total_energy_in_c(positions, n_beads)
    # Placeholder example
    energy = 0.0  # Replace with actual call to C code
    return energy

# Optimized pairwise distance computation (using C for efficiency)
def pairwise_distance_optimized(positions, n_beads, cell_size=1.0):
    # Step 1: Partition the 3D space into cells
    cells = {}
    for i in range(n_beads):
        x, y, z = positions[i]
        cell_coords = (int(x // cell_size), int(y // cell_size), int(z // cell_size))
        if cell_coords not in cells:
            cells[cell_coords] = []
        cells[cell_coords].append(i)

    # Step 2: Calculate pairwise distances only for beads in the same or neighboring cells
    distances = np.zeros((n_beads, n_beads))
    for cell_coords, bead_indices in cells.items():
        for i in range(len(bead_indices)):
            for j in range(i + 1, len(bead_indices)):
                bead_i = bead_indices[i]
                bead_j = bead_indices[j]
                r = np.linalg.norm(positions[bead_i] - positions[bead_j])
                distances[bead_i, bead_j] = distances[bead_j, bead_i] = r

            # Check neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        neighbor_coords = (cell_coords[0] + dx, cell_coords[1] + dy, cell_coords[2] + dz)
                        if neighbor_coords in cells:
                            for bead_k in cells[neighbor_coords]:
                                for bead_i in bead_indices:
                                    r = np.linalg.norm(positions[bead_i] - positions[bead_k])
                                    distances[bead_i, bead_k] = distances[bead_k, bead_i] = r
    return distances

# Optimization function (modified for performance)
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    trajectory = []

    # Callback for tracking progress
    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(f"Iteration {len(trajectory)}: Progress")

    # Perform optimization using L-BFGS (Limited-memory BFGS)
    result = minimize(
        fun=total_energy_optimized,  # Optimized energy function
        x0=positions.flatten(),
        args=(n_beads,),  # Pass necessary parameters
        method='L-BFGS-B',  # Using L-BFGS-B instead of BFGS for large-scale problems
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )

    # Check if trajectory contains any data before attempting to save
    if trajectory:
        if write_csv:
            csv_filepath = f'protein{n_beads}.csv'
            print(f'Writing data to file {csv_filepath}')
            np.savetxt(csv_filepath, trajectory[-1], delimiter=",")
    else:
        print("No intermediate trajectories available for saving.")

    return result, trajectory


# 3D Visualization
def plot_protein_3d(positions, title="Protein Conformation", ax=None):
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

# Animation function
def animate_optimization(trajectory, interval=100):
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

    ani = FuncAnimation(
        fig, update, frames=len(trajectory), interval=interval, blit=False
    )
    plt.show()

# Main function for running the optimization
if __name__ == "__main__":
    for n_beads in [10, 100, 500]:
        initial_positions = initialize_protein(n_beads)

        # Now using total_energy_optimized for energy calculation
        print(f"Initial Energy for {n_beads} beads:", total_energy_optimized(initial_positions.flatten(), n_beads))
        plot_protein_3d(initial_positions, title=f"Initial Configuration - {n_beads} Beads")

        result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)

        optimized_positions = result.x.reshape((n_beads, 3))
        print(f"Optimized Energy for {n_beads} beads:", total_energy_optimized(optimized_positions.flatten(), n_beads))
        plot_protein_3d(optimized_positions, title=f"Optimized Configuration - {n_beads} Beads")

        # Animate the optimization process
        animate_optimization(trajectory)
