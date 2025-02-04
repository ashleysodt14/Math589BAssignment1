import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import ctypes
from ctypes import c_double, POINTER

# Load C library
libenergy = ctypes.CDLL('./libenergy.so')  # Ensure this is compiled correctly

# Set argument and return types for C function
libenergy.total_energy.argtypes = [POINTER(c_double), ctypes.c_int, c_double, c_double, c_double, c_double]
libenergy.total_energy.restype = c_double

# Initialize protein positions
def initialize_protein(n_beads, dimension=3, fudge=1e-5):
    positions = np.zeros((n_beads, dimension))
    for i in range(1, n_beads):
        positions[i, 0] = positions[i-1, 0] + 1  # Fixed bond length of 1 unit
        positions[i, 1] = fudge * np.sin(i)
        positions[i, 2] = fudge * np.sin(i*i)
    return positions

# Total energy function using C-optimized energy calculation
def total_energy(positions, n_beads, epsilon=1.0, sigma=1.0, b=1.0, k_b=100.0):
    positions_flat = positions.flatten()
    energy = libenergy.total_energy(
        (c_double * len(positions_flat))(*positions_flat), n_beads, epsilon, sigma, b, k_b
    )
    return energy

# Optimization function
def optimize_protein(positions, n_beads, write_csv=False, maxiter=1000, tol=1e-6):
    trajectory = []

    def callback(x):
        trajectory.append(x.reshape((n_beads, -1)))
        if len(trajectory) % 20 == 0:
            print(len(trajectory))

    result = minimize(
        fun=total_energy,
        x0=positions.flatten(),
        args=(n_beads,),
        method='BFGS',
        callback=callback,
        tol=tol,
        options={'maxiter': maxiter, 'disp': True}
    )
    
    if write_csv:
        csv_filepath = f'protein{n_beads}.csv'
        print(f'Writing data to file {csv_filepath}')
        np.savetxt(csv_filepath, trajectory[-1], delimiter=",")

    return result, trajectory

# 3D visualization function
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

# Animation function with autoscaling
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

# Main function
if __name__ == "__main__":
    n_beads = 10
    dimension = 3
    initial_positions = initialize_protein(n_beads, dimension)

    print("Initial Energy:", total_energy(initial_positions.flatten(), n_beads))
    plot_protein_3d(initial_positions, title="Initial Configuration")

    result, trajectory = optimize_protein(initial_positions, n_beads, write_csv=True)

    optimized_positions = result.x.reshape((n_beads, dimension))
    print("Optimized Energy:", total_energy(optimized_positions.flatten(), n_beads))
    plot_protein_3d(optimized_positions, title="Optimized Configuration")

    # Animate the optimization process
    animate_optimization(trajectory)
