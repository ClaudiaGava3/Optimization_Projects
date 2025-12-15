import numpy as np
import matplotlib.pyplot as plt
from mpc import run_mpc_simulation

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "single"  # Cambia in "double" se vuoi testare il doppio pendolo
N_TRIALS = 20          # Numero prove
M_SHORT = 20           # Orizzonte Corto (M)
N_LONG = 100           # Orizzonte Lungo (N)
SIM_STEPS = 100        # Durata simulazione

print(f"--- INIZIO COMPARAZIONE (Robot: {ROBOT_TYPE}) ---")

# A: corto senza rete
# B: corto con rete
# C: lungo senza rete

costs = {'A': [], 'B': [], 'C': []}
successes = {'A': 0, 'B': 0, 'C': 0}

# Generazione stati iniziali casuali
np.random.seed(42) # Per riproducibilità
initial_states = []
for _ in range(N_TRIALS):
    # Genera angolo casuale (evitando lo zero esatto)
    q_rnd = np.random.uniform(-np.pi, np.pi)
    dq_rnd = np.random.uniform(-1.0, 1.0)
    
    if ROBOT_TYPE == "single":
        x0 = np.array([q_rnd, dq_rnd])
    else:
        # Per il doppio pendolo, aggiungiamo altre 2 dimensioni
        x0 = np.array([q_rnd, 0.0, dq_rnd, 0.0])
    
    initial_states.append(x0)

# --- LOOP DI TEST ---
for i, x0 in enumerate(initial_states):
    print(f"\nTrial {i+1}/{N_TRIALS}...")
    
    # 1. CASO A: Orizzonte Corto (M), No Rete
    print("   Running Case A (Short, NoNet)...")
    c_a, _, _, ok= run_mpc_simulation(ROBOT_TYPE, horizon=M_SHORT, use_network=False, x_init=x0, sim_steps=SIM_STEPS)
    costs['A'].append(c_a)
    if ok: successes['A'] += 1

    # 2. CASO B: Orizzonte Corto (M), CON Rete -> Il tuo metodo
    print("   Running Case B (Short, Net)...")
    c_b, _, _, ok= run_mpc_simulation(ROBOT_TYPE, horizon=M_SHORT, use_network=True, x_init=x0, sim_steps=SIM_STEPS)
    costs['B'].append(c_b)
    if ok: successes['B'] += 1

    # 3. CASO C: Orizzonte Lungo (N), No Rete -> Benchmark
    print("   Running Case C (Long, NoNet)...")
    c_c, _, _, ok = run_mpc_simulation(ROBOT_TYPE, horizon=N_LONG, use_network=False, x_init=x0, sim_steps=SIM_STEPS)
    costs['C'].append(c_c)
    if ok: successes['C'] += 1


# Ottima idea. Monitorare la Percentuale di Successo è fondamentale per il report, perché il "Costo Medio" può essere ingannevole (se un metodo fallisce sempre ma con costo basso perché sta fermo, sembra buono ma non lo è).

print("\n--- RISULTATI ---")
labels = ['A (M)', 'B (M + Net)', 'C (N)']
avg_costs = [np.mean(costs['A']), np.mean(costs['B']), np.mean(costs['C'])]
succ_rates = [successes['A'], successes['B'], successes['C']]

print(f"Successi su {N_TRIALS}: A={succ_rates[0]}, B={succ_rates[1]}, C={succ_rates[2]}")
print(f"Costi Medi: A={avg_costs[0]:.0f}, B={avg_costs[1]:.0f}, C={avg_costs[2]:.0f}")

# --- PLOT 1: GRAFICO A BARRE COSTI ---
plt.figure(figsize=(8, 5))
plt.bar(labels, avg_costs, color=['red', 'green', 'blue'], alpha=0.7)
plt.ylabel('Costo Medio Accumulato')
plt.title('Confronto Costi MPC')
plt.grid(axis='y', linestyle='--')
plt.savefig(f"comparison_costs_{ROBOT_TYPE}.png")

# --- PLOT 2: GRAFICI A TORTA SUCCESSI ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Tasso di Successo su {N_TRIALS} prove ({ROBOT_TYPE})', fontsize=16)

colors = ['#66b3ff', '#ff9999'] # Blu (Successo), Rosso (Fallimento)
explode = (0.1, 0)  

for idx, method in enumerate(['A', 'B', 'C']):
    succ = successes[method]
    fail = N_TRIALS - succ
    
    axs[idx].pie([succ, fail], labels=['Success', 'Fail'], autopct='%1.1f%%', 
                 startangle=90, colors=['mediumseagreen', 'salmon'], explode=explode)
    axs[idx].set_title(f'Metodo {labels[idx]}')

plt.tight_layout()

plt.show()