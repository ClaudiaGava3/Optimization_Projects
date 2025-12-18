import numpy as np
import matplotlib.pyplot as plt
from optimal_control.casadi_adam.final_project_.mpc.mpc_funzione import run_mpc_simulation

# --- IMPORT MODELLI ---
try:
    from optimal_control.casadi_adam.final_project_.models.pendulum_model import PendulumModel
    from optimal_control.casadi_adam.final_project_.models.doublependulum_model import DoublePendulumModel
except ImportError:
    print("ERRORE: Modelli non trovati.")
    exit()

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "double"  # "double" o "single"
M_SHORT = 20           # Orizzonte Corto
N_LONG = 100           # Orizzonte Lungo
SIM_STEPS = 300        # Tempo simulazione
DT = 0.01

print(f"--- GENERAZIONE GRAFICI COMPARATIVI: {ROBOT_TYPE} ---")

# 1. Stato Iniziale "Difficile" (Swing-Up da fermo)
if ROBOT_TYPE == "single":
    robot = PendulumModel()
    x_init = np.array([0.0, 0.0])  # Giù fermo
    q_target = np.array([np.pi])
    ylims_q = [-1, 4]
elif ROBOT_TYPE == "double":
    robot = DoublePendulumModel()
    x_init = np.array([0.0, 0.0, 0.0, 0.0]) # Giù fermo
    q_target = np.array([np.pi, 0.0])
    ylims_q = [-7, 7]
print(f"Testando Swing-Up da: {x_init}")

# 2. Eseguo le 3 simulazioni
print("Simulazione A (M, No NN)...")
_, _, _, ok_a, x_a, u_a = run_mpc_simulation(ROBOT_TYPE, M_SHORT, False, x_init, SIM_STEPS)

print("Simulazione B (M, Neural)...")
_, _, _, ok_b, x_b, u_b = run_mpc_simulation(ROBOT_TYPE, M_SHORT, True, x_init, SIM_STEPS)

print("Simulazione C (Benchmark)...")
_, _, _, ok_c, x_c, u_c = run_mpc_simulation(ROBOT_TYPE, N_LONG+M_SHORT, False, x_init, SIM_STEPS)

# 3. Creazione Grafici Sovrapposti
nq = robot.nq
time = np.arange(len(x_a)) * DT

# Creo una figura unica con sottotrame
fig, axs = plt.subplots(3, nq, figsize=(12, 10), sharex=True)
if nq == 1: axs = axs.reshape(3, 1) # Fix dimensioni array per pendolo singolo

# Colori e Stili
# A = Rosso Tratteggiato
# B = Verde Solido
# C = Blu Puntinato
style_a = {'color': 'red', 'linestyle': '--', 'label': f'Case A (M={M_SHORT})', 'alpha': 0.7}
style_b = {'color': 'green', 'linestyle': '-', 'label': f'Case B (Neural M={M_SHORT})', 'linewidth': 2}
style_c = {'color': 'blue', 'linestyle': ':', 'label': f'Case C (Bench N+M={N_LONG+M_SHORT})', 'linewidth': 2, 'alpha': 0.8}

# --- PLOT POSIZIONI ---
for j in range(nq):
    ax = axs[0, j]
    ax.plot(time, x_a[:, j], **style_a)
    ax.plot(time, x_c[:, j], **style_c) # C sotto B
    ax.plot(time, x_b[:, j], **style_b) # B sopra per evidenziarlo
    
    # Target
    ax.axhline(q_target[j], color='gray', linestyle='-.', label='Target')
    
    ax.set_ylabel(f'Pos q_{j+1} [rad]')
    ax.set_title(f'Joint {j+1} Position')
    ax.grid(True)
    if j == 0: ax.legend(loc='lower right')

# --- PLOT VELOCITÀ ---
for j in range(nq):
    ax = axs[1, j]
    ax.plot(time, x_a[:, nq+j], **style_a)
    ax.plot(time, x_c[:, nq+j], **style_c)
    ax.plot(time, x_b[:, nq+j], **style_b)
    
    ax.set_ylabel(f'Vel dq_{j+1} [rad/s]')
    ax.grid(True)

# --- PLOT COPPIE ---
# u è più corto di x di 1, accorciamo time
t_u = time[:-1]
for j in range(nq):
    ax = axs[2, j]
    ax.plot(t_u, u_a[:, j], **style_a)
    ax.plot(t_u, u_c[:, j], **style_c)
    ax.plot(t_u, u_b[:, j], **style_b)
    
    ax.set_ylabel(f'Torque u_{j+1} [Nm]')
    ax.set_xlabel('Time [s]')
    ax.grid(True)

plt.suptitle(f"Confronto Traiettorie: Swing-Up ({ROBOT_TYPE})", fontsize=16)
plt.tight_layout()
filename = f"trajectory_comparison_{ROBOT_TYPE}.png"
plt.savefig(filename)
print(f"Grafico salvato come {filename}")
plt.show()