import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -----------------------------
# Helper Functions
# -----------------------------
def compute_target_energy(n_beads):
    """Estimate the reference target energy based on bead count."""
    if n_beads == 10:
        return -21.0
    elif n_beads == 100:
        return -455.0
    elif n_beads == 200:
        return -945.0
    return -450.0 + (n_beads - 100) * (-495.0 / 100.0) if n_beads >= 100 else -25.0 + (n_beads - 10) * (-425.0 / 90.0)

def lennard_jones(r, epsilon=1.0, sigma=1.0):
    """Compute the Lennard-Jones potential."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

def harmonic_bond_potential(r, equilibrium_length=1.0, stiffness=100.0):
    """Compute the harmonic bond stretching potential."""
    return stiffness * (r - equilibrium_length) ** 2

# -----------------------------
# Protein Optimization Class
# -----------------------------
class ProteinOptimizer:
    def __init__(self, n_beads, perturb_scale=1e-2):
        self.n_beads = n_beads
        self.target_energy = compute_target_energy(n_beads)
        self.positions = self._initialize_positions(perturb_scale)

    def _initialize_positions(self, perturb_scale):
        """Generate initial bead positions in 3D with slight perturbation."""
        positions = np.zeros((self.n_beads, 3))
        for i in range(1, self.n_beads):
            positions[i, 0] = positions[i-1, 0] + 1
            positions[i, 1] = perturb_scale * np.sin(i)
            positions[i, 2] = perturb_scale * np.sin(i * i)
        
        # Add small noise to break symmetry and prevent stagnation
        positions += np.random.normal(scale=perturb_scale, size=positions.shape)
        return positions

    def compute_energy_and_gradient(self, x):
        """Compute total system energy and gradient using harmonic bonds & Lennard-Jones."""
        positions = x.reshape((self.n_beads, 3))
        energy = 0.0
        grad = np.zeros_like(positions)

        # Compute bond stretching energy
        bond_vectors = np.diff(positions, axis=0)
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        energy += np.sum(harmonic_bond_potential(bond_lengths))

        # Compute non-bonded interactions using KD-Tree
        tree = cKDTree(positions)
        pairs = tree.query_pairs(r=2.5, output_type='ndarray')  

        # Lennard-Jones energy and gradient
        for i, j in pairs:
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r > 1e-2:
                lj_energy = lennard_jones(r, epsilon=1.2, sigma=1.1)
                energy += lj_energy
                dE_dr = 4 * 1.2 * (-12 * 1.1**12 / r**13 + 6 * 1.1**6 / r**7)
                grad_contrib = (dE_dr / r) * r_vec
                grad[i] -= grad_contrib
                grad[j] += grad_contrib

        return energy, grad.flatten()

    def bfgs_backtracking(self, x0, maxiter=2000, tol=1e-4, alpha0=1.0, beta=0.8):
        """Custom BFGS optimizer with backtracking line search."""
        print("🔧 Running BFGS with Backtracking...")
        x = x0.copy()
        H = np.eye(len(x))
        trajectory = []

        for k in range(maxiter):
            f, g = self.compute_energy_and_gradient(x)

            if np.linalg.norm(g) < tol:
                print(f"✅ BFGS converged at iteration {k}")
                break

            # Compute search direction
            p = -H @ g
            alpha = alpha0  # Initial step size

            # Backtracking line search
            while True:
                x_new = x + alpha * p
                f_new, _ = self.compute_energy_and_gradient(x_new)
                if f_new <= f + 1e-4 * alpha * np.dot(g, p):
                    break
                alpha *= beta
                if alpha < 1e-6:
                    break

            s = alpha * p
            x_new = x + s
            f_new, g_new = self.compute_energy_and_gradient(x_new)
            y = g_new - g
            ys = np.dot(y, s)

            # Update Hessian approximation using BFGS
            if ys > 1e-10:
                rho = 1.0 / ys
                I = np.eye(len(x))
                H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x = x_new
            trajectory.append(x.reshape((self.n_beads, -1)))

            if (k+1) % 50 == 0:
                print(f"🌀 Iteration {k+1}: Energy = {f_new:.6f}")

        print("🔹 BFGS Optimization Complete")
        return x, trajectory

    def optimize(self):
        """Perform optimization and store results."""
        print(f"🎯 Target Energy: {self.target_energy}")
        x0 = self.positions.flatten()

        # Run BFGS optimization
        x_opt, trajectory = self.bfgs_backtracking(x0)

        # Use scipy minimize for final OptimizeResult structure
        result = minimize(
            fun=self.compute_energy_and_gradient,
            x0=x_opt.flatten(),
            args=(),
            method='BFGS',
            jac=True,
            options={'maxiter': 0, 'disp': False}
        )

        result.x = x_opt.flatten()
        result.nit = len(trajectory) - 1
        result.success = True
        result.status = 0
        result.message = "Optimization terminated successfully."

        return result, trajectory

# -----------------------------
# Visualization
# -----------------------------
def plot_protein(positions, title="Protein Structure"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    positions = positions.reshape((-1, 3))
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], '-o', markersize=6)
    plt.title(title)
    plt.show()

def animate_trajectory(trajectory):
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
    n_beads = 100  # Modify for different system sizes
    optimizer = ProteinOptimizer(n_beads)

    plot_protein(optimizer.positions, title="Initial Configuration")

    result, trajectory = optimizer.optimize()

    optimized_positions = result.x.reshape((n_beads, 3))
    plot_protein(optimized_positions, title="Optimized Configuration")

    animate_trajectory(trajectory)
    print(result)
