import numpy as np
import torch
import multiprocessing
import time
import os
import sys
import casadi as ca
from pendulum_system import build_parametric_ocp

# --- Configurazione ---
N_SAMPLES = 10000
N_HORIZON = 60
# MODIFICA: Usiamo metà dei core per evitare crash di memoria (es. 12 -> 6)
N_CORES = max(1, int(multiprocessing.cpu_count() / 2)) 

# Globali Worker
worker_opti = None
worker_param = None
worker_X = None
worker_U = None
worker_J = None

# --- Dinamica NumPy Veloce (per Initial Guess) ---
def numpy_dynamics(x, u, dt=0.02):
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    q1, q2, dq1, dq2 = x
    tau1, tau2 = u
    M11 = (m1 + m2) * l1**2
    M12 = m2 * l1 * l2 * np.cos(q1 - q2)
    M22 = m2 * l2**2
    det = M11*M22 - M12*M12
    C1 = -m2 * l1 * l2 * dq2**2 * np.sin(q1 - q2)
    C2 =  m2 * l1 * l2 * dq1**2 * np.sin(q1 - q2)
    G1 = (m1 + m2) * g * l1 * np.sin(q1)
    G2 = m2 * g * l2 * np.sin(q2)
    rhs1 = tau1 - C1 - G1
    rhs2 = tau2 - C2 - G2
    ddq1 = (M22 * rhs1 - M12 * rhs2) / det
    ddq2 = (M11 * rhs2 - M12 * rhs1) / det
    return np.array([dq1, dq2, ddq1, ddq2])

def rk4_step(x, u, dt=0.02):
    k1 = numpy_dynamics(x, u)
    k2 = numpy_dynamics(x + 0.5*dt*k1, u)
    k3 = numpy_dynamics(x + 0.5*dt*k2, u)
    k4 = numpy_dynamics(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_trajectory_guess(x0, horizon):
    traj = np.zeros((4, horizon + 1))
    traj[:, 0] = x0
    u_zero = np.array([0.0, 0.0])
    x_curr = x0.copy()
    for i in range(horizon):
        x_curr = rk4_step(x_curr, u_zero)
        traj[:, i+1] = x_curr
    return traj

# --- Worker Logic ---
def init_worker(horizon):
    global worker_opti, worker_param, worker_X, worker_U, worker_J
    sys.stdout = open(os.devnull, 'w')
    worker_opti, worker_param, worker_X, worker_U, worker_J = build_parametric_ocp(horizon)
    sys.stdout = sys.__stdout__

def solve_single_sample(seed):
    np.random.seed(seed)
    # Goal Biased Sampling
    if np.random.rand() < 0.2:
        q1 = np.pi + np.random.uniform(-0.5, 0.5)
        q2 = 0.0   + np.random.uniform(-0.5, 0.5)
        dq1 = np.random.uniform(-1.0, 1.0)
        dq2 = np.random.uniform(-1.0, 1.0)
    else:
        q1 = np.random.uniform(0, 2*np.pi)
        q2 = np.random.uniform(-np.pi, np.pi)
        dq1 = np.random.uniform(-6.0, 6.0)
        dq2 = np.random.uniform(-6.0, 6.0)
    x0 = np.array([q1, q2, dq1, dq2])
    
    try:
        x_guess = simulate_trajectory_guess(x0, N_HORIZON)
        worker_opti.set_value(worker_param, x0)
        worker_opti.set_initial(worker_X, x_guess)
        worker_opti.set_initial(worker_U, 0.0)
        sol = worker_opti.solve()
        return (x0, sol.value(worker_J))
    except:
        return None

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"--- Generazione Dataset High-Speed (Safe Mode) ---")
    print(f"Campioni Totali: {N_SAMPLES}")
    print(f"Cores: {N_CORES} (Dimezzati per stabilità memoria)")
    print("-" * 60)
    
    start_time = time.time()
    seeds = np.random.randint(0, 1000000, N_SAMPLES)
    results = []
    
    with multiprocessing.Pool(processes=N_CORES, initializer=init_worker, initargs=(N_HORIZON,)) as pool:
        iterator = pool.imap_unordered(solve_single_sample, seeds)
        for i, res in enumerate(iterator):
            results.append(res)
            if (i + 1) % 50 == 0 or (i + 1) == N_SAMPLES:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                percent = (i + 1) / N_SAMPLES * 100
                remaining = (N_SAMPLES - (i+1)) / speed if speed > 0 else 0
                sys.stdout.write(f"\rProgresso: {i+1}/{N_SAMPLES} [{percent:.1f}%] | Vel: {speed:.1f} it/s | ETA: {remaining:.0f}s  ")
                sys.stdout.flush()
    
    print("\n" + "-" * 60)
    valid_data = [r for r in results if r is not None]
    X_data = np.array([r[0] for r in valid_data])
    Y_data = np.array([r[1] for r in valid_data]).reshape(-1, 1)
    
    print(f"Dataset salvato: {len(valid_data)} samples validi.")
    torch.save({'X': torch.tensor(X_data, dtype=torch.float32),
                'Y': torch.tensor(Y_data, dtype=torch.float32)}, 
               'dataset_double_pendulum.pt')

if __name__ == "__main__":
    main()