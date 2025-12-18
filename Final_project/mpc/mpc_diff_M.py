import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# --- IMPORTAZIONI FILE ---
from optimal_control.casadi_adam.final_project_.mpc.mpc_funzione import run_mpc_simulation
from optimal_control.casadi_adam.final_project_.models.pendulum_model import PendulumModel
from optimal_control.casadi_adam.final_project_.models.doublependulum_model import DoublePendulumModel
from optimal_control.casadi_adam.final_project_.ocp.random_generator import generate_random_initial_states
from optimal_control.casadi_adam.final_project_.plot.plot import animate_pendulum, animate_double_pendulum

# --- CONFIGURAZIONE ---
# Robot "single" o "double"
ROBOT_TYPE = "double" 

# Lista degli orizzonti da testare
M_VALUES = [20, 15, 10, 7, 6, 5]

# Altri parametri
DT = 0.01
SIM_STEPS = 300  # 3.0 secondi
SUCCESS_THRESHOLD = 0.15

# --- 1. SETUP INIZIALE ---
print(f"--- ANALISI COMPARATIVA ORIZZONTI (M) - Robot: {ROBOT_TYPE} ---")

if ROBOT_TYPE == "single":
    robot = PendulumModel()
    q_des = np.array([np.pi])
elif ROBOT_TYPE == "double":
    robot = DoublePendulumModel()
    q_des = np.array([np.pi, 0.0])

nq = robot.nq

# Generazione Stato Iniziale FISSO per tutti i test
x_init = generate_random_initial_states(robot, n_samples=1)[0]

print(f"Stato Iniziale Fisso: {x_init}")

results = {}

# --- 2. LOOP DI SIMULAZIONE ---
for M in M_VALUES:
    print(f"\nTestando Horizon M = {M}...")
    
    cost, mse, err, success, x_hist, u_hist = run_mpc_simulation(
        robot_type=ROBOT_TYPE,
        horizon=M,
        use_network=True,
        x_init=x_init,
        sim_steps=SIM_STEPS,
        dt=DT
    )
    
    print(f"   -> Esito: {'✅ Success' if success else '❌ Fail'}")
    print(f"   -> Errore Finale: {err:.4f}")
    
    results[M] = {
        'x': x_hist,
        'u': u_hist,
        'error': err,
        'success': success
    }

# --- 3. PLOTTING COMPARATIVO ---
print("\nGenerazione Grafici...")
colors = plt.cm.viridis(np.linspace(0, 0.9, len(M_VALUES)))
time_axis = np.arange(SIM_STEPS + 1) * DT

# A. POSIZIONI
plt.figure(figsize=(12, 8))
for idx, M in enumerate(M_VALUES):
    res = results[M]
    t_curr = np.arange(len(res['x'])) * DT
    
    for j in range(nq):
        lbl = f'M={M}' if j == 0 else None
        ls = '-' if j==0 else '--'
        plt.plot(t_curr, res['x'][:, j], color=colors[idx], linestyle=ls, linewidth=2, label=lbl)

# Target lines
for j in range(nq):
    plt.axhline(q_des[j], color='red', linestyle=':', alpha=0.5, label='Target' if j==0 else None)

plt.title(f"Position Comparation due to M ({ROBOT_TYPE})")
plt.xlabel("Time [s]")
plt.ylabel("Position [rad]")
plt.legend()
plt.grid(True)
plt.show()

# B. VELOCITÀ
plt.figure(figsize=(12, 6))
for idx, M in enumerate(M_VALUES):
    res = results[M]
    t_curr = np.arange(len(res['x'])) * DT
    for j in range(nq):
        ls = '-' if j==0 else '--'
        plt.plot(t_curr, res['x'][:, nq+j], color=colors[idx], linestyle=ls, label=f'M={M}' if j==0 else None)

plt.title(f"Velocity Comparation due to M ({ROBOT_TYPE})")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [rad/s]")
plt.legend()
plt.grid(True)
plt.show()

# C. INPUT (COPPIE)
plt.figure(figsize=(12, 6))
for idx, M in enumerate(M_VALUES):
    res = results[M]
    t_curr = np.arange(len(res['u'])) * DT
    
    # Gestione shape u (se 1D o 2D)
    u_data = res['u']
    if u_data.ndim == 1: u_data = u_data.reshape(-1, 1)
    
    for j in range(nq):
        ls = '-' if j==0 else '--'
        plt.plot(t_curr, u_data[:, j], color=colors[idx], linestyle=ls, label=f'M={M}' if j==0 else None)

plt.title("Torque Comparation due to M ({ROBOT_TYPE})")
plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend()
plt.grid(True)
plt.show()

# --- 4. ANIMAZIONE BEST CASE ---
# Trovo M con errore minimo
best_M = min(results, key=lambda k: results[k]['error'])
print(f"\n--- ANIMAZIONE BEST CASE: M={best_M} (Err: {results[best_M]['error']:.4f}) ---")

best_x = results[best_M]['x']
q_anim = best_x[:, :nq].T
t_anim = np.arange(len(best_x)) * DT

if ROBOT_TYPE == "single":
    animate_pendulum(t_anim, q_anim[0, :], length=robot.l, dt=DT)
elif ROBOT_TYPE == "double":
    animate_double_pendulum(t_anim, q_anim, l1=robot.l1, l2=robot.l2, dt=DT)