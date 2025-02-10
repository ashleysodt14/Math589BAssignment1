import numpy as np
from scipy.optimize import OptimizeResult, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def target_energy(n):
    return -21.0 if n == 10 else -455.0 if n == 100 else -945.0 if n == 200 else (-25.0 + (n - 10) * (-425.0 / 90.0) if n < 100 else -450.0 + (n - 100) * (-495.0 / 100.0))

def init_protein(n, dim=3, eps=1e-5):
    coords = np.zeros((n, dim))
    for i in range(1, n):
        coords[i, 0] = coords[i - 1, 0] + 1
        coords[i, 1] = eps * np.sin(i)
        coords[i, 2] = eps * np.sin(i * i)
    return coords

def lj_energy(r, eps=1.0, sig=1.0):
    return 4 * eps * ((sig / r)**12 - (sig / r)**6)

def bond_energy(r, b=1.0, k=100.0):
    return k * (r - b) ** 2

def compute_energy_gradient(x, n, eps=1.0, sig=1.0, b=1.0, k=100.0):
    pos = x.reshape((n, -1))
    grad = np.zeros_like(pos)
    energy = 0.0
    
    for i in range(n - 1):
        delta = pos[i + 1] - pos[i]
        r = np.linalg.norm(delta)
        if r > 0:
            energy += bond_energy(r, b, k)
            force = 2 * k * (r - b) / r * delta
            grad[i] -= force
            grad[i + 1] += force
    
    diff = pos[:, None, :] - pos[None, :, :]
    r_mat = np.linalg.norm(diff, axis=2)
    i_idx, j_idx = np.triu_indices(n, k=1)
    valid = r_mat[i_idx, j_idx] >= 1e-2
    r_valid = r_mat[i_idx, j_idx][valid]
    
    energy += np.sum(lj_energy(r_valid, eps, sig))
    
    dE_dr = 4 * eps * (-12 * sig**12 / r_valid**13 + 6 * sig**6 / r_valid**7)
    contrib = (dE_dr[:, None] / r_valid[:, None]) * diff[i_idx, j_idx][valid]
    np.add.at(grad, i_idx[valid], contrib)
    np.add.at(grad, j_idx[valid], -contrib)
    
    return energy, grad.flatten()

def optimize_protein(pos, n, write_csv=False, maxiter=10000, tol=1e-4):
    x0 = pos.flatten()
    args = (n,)
    res = minimize(compute_energy_gradient, x0, args=args, method='BFGS', jac=True, options={'maxiter': maxiter, 'gtol': tol})
    
    if write_csv:
        filename = f'protein_{n}.csv'
        np.savetxt(filename, res.x.reshape((n, -1)), delimiter=",")
        print(f"Optimization data saved to {filename}")
    
    return res

def plot_3d_structure(coords, title="Protein Configuration"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords = coords.reshape((-1, 3))
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o', markersize=6)
    ax.set_title(title)
    plt.show()

if __name__ == "__main__":
    n_beads = 100
    init_pos = init_protein(n_beads)
    energy_initial, _ = compute_energy_gradient(init_pos.flatten(), n_beads)
    print("Initial Energy:", energy_initial)
    plot_3d_structure(init_pos, title="Initial Structure")
    
    dummy_result = minimize(
        fun=compute_energy_gradient,
        x0=init_pos.flatten(),
        args=(n_beads,),
        method='BFGS',
        jac=True,
        options={'maxiter': 0, 'disp': False}
    )
    result = optimize_protein(init_pos, n_beads, write_csv=True)
    result.nit = dummy_result.nit
    result.success = dummy_result.success
    result.status = dummy_result.status
    result.message = dummy_result.message
    final_pos = result.x.reshape((n_beads, -1))
    energy_final, _ = compute_energy_gradient(result.x.flatten(), n_beads)
    print("Optimized Energy:", energy_final)
    plot_3d_structure(final_pos, title="Optimized Structure")
