import numpy as np
import matplotlib.pyplot as plt
from final_project_.mpc.mpc_funzione import run_mpc_simulation
from time import time as clock

# --- IMPORTO IL GENERATORE RANDOM ---
try:
    from final_project_.ocp.random_generator import generate_random_initial_states
    from final_project_.models.pendulum_model import PendulumModel
    from final_project_.models.doublependulum_model import DoublePendulumModel
except ImportError:
    print("ERRORE: random_generator.py o modelli non trovati.")
    exit()

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "double"  # "single" o "double"
N_TRIALS = 100         # Numero prove random
#M_SHORT = 5           # Orizzonte Corto per single (quello usato con successo)
M_SHORT = 20           # Orizzonte Corto per double
N_LONG = 100           # Orizzonte Lungo
SIM_STEPS = 300        # Durata simulazione (step) per dare tempo di arrivare

print(f"--- INIZIO COMPARAZIONE (Robot: {ROBOT_TYPE}) ---")
print(f"Horizon Short: {M_SHORT}, Horizon Long: {N_LONG}")
print(f"Trials: {N_TRIALS}")

# Setup robot temporaneo per generare stati
if ROBOT_TYPE == "single": temp_robot = PendulumModel()
else: temp_robot = DoublePendulumModel()

# --- GENERAZIONE STATI INIZIALI ---
initial_states = generate_random_initial_states(temp_robot, n_samples=N_TRIALS)

# Strutture dati per risultati
# Costi (più basso è meglio)
costs = {'A': [], 'B': [], 'C': []}
# Errori MSE rispetto al target
errors_mse = {'A': [], 'B': [], 'C': []}
# Tassi di successo
success_flags = {'A': [], 'B': [], 'C': []}

times = {'A': [], 'B': [], 'C': []}

# --- LOOP DI TEST ---
for i in range(N_TRIALS):
    x0 = initial_states[i]
    print(f"\n>>> Trial {i+1}/{N_TRIALS} - Init: {x0}")
    
    # CASO A
    t_start = clock() # START
    c_a, m_a, _, ok_a, x_h_a, u_h_a = run_mpc_simulation(ROBOT_TYPE, M_SHORT, False, x0, SIM_STEPS)
    t_end = clock()   # END
    costs['A'].append(c_a); errors_mse['A'].append(m_a); success_flags['A'].append(ok_a)
    times['A'].append(t_end - t_start) # Salva tempo

    # CASO B
    t_start = clock()
    c_b, m_b, _, ok_b, x_h_b, u_h_b = run_mpc_simulation(ROBOT_TYPE, M_SHORT, True, x0, SIM_STEPS)
    t_end = clock()
    costs['B'].append(c_b); errors_mse['B'].append(m_b); success_flags['B'].append(ok_b)
    times['B'].append(t_end - t_start)

    # CASO C
    t_start = clock()
    c_c, m_c, _, ok_c, x_h_c, u_h_c = run_mpc_simulation(ROBOT_TYPE, N_LONG+M_SHORT, False, x0, SIM_STEPS)
    t_end = clock()
    costs['C'].append(c_c); errors_mse['C'].append(m_c); success_flags['C'].append(ok_c)
    times['C'].append(t_end - t_start)

# --- ANALISI DATI ---
def get_stats(data_list):
    return np.mean(data_list), np.std(data_list)

print("\n" + "="*40)
print("   REPORT FINALE")
print("="*40)

methods = [('A', f'Short (M={M_SHORT})'), 
           ('B', f'Neural (M={M_SHORT})'), 
           ('C', f'Long (N+M={N_LONG+M_SHORT})')]

# Calcolo percentuali successo
succ_rates = {}
for k, label in methods:
    succ_rate = np.sum(success_flags[k]) / N_TRIALS * 100
    succ_rates[k] = succ_rate
    
    # Calcolo costo medio SOLO sulle prove di successo (per non falsare con costi infiniti)
    # Oppure su tutte se voglio penalizzare i fallimenti
    valid_costs = [c for c, ok in zip(costs[k], success_flags[k])]
    mean_c = np.mean(costs[k])
    
    mean_mse = np.mean(errors_mse[k])

    avg_time = np.mean(times[k])
    
    print(f"Metodo {k} [{label}]:")
    print(f"  Success Rate: {succ_rate:.1f}%")
    print(f"  Avg Cost:     {mean_c:.2f}")
    print(f"  Avg MSE:      {mean_mse:.4f}")
    print(f"  Avg Time:     {avg_time:.4f} s")
    print("-" * 20)
    

# --- PLOTTING ---
labels = [m[1] for m in methods]
keys = [m[0] for m in methods]

# 1. Tasso di Successo (Bar Chart)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
rates = [succ_rates[k] for k in keys]
bars = plt.bar(keys, rates, color=['gray', 'orange', 'green'], alpha=0.7)
plt.ylabel('Success Rate (%)')
plt.title(f'Reliability ({ROBOT_TYPE})')
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)
# Aggiungo etichette sulle barre
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.0f}%", ha='center', va='bottom')

# 2. Costo Medio (Box Plot per vedere la varianza)
plt.subplot(1, 2, 2)
data_to_plot = [costs['A'], costs['B'], costs['C']]
plt.boxplot(data_to_plot, labels=keys, patch_artist=True, 
            boxprops=dict(facecolor="lightblue"))
plt.ylabel('Total Cost (Accumulated)')
plt.title('Cost Distribution')
plt.grid(linestyle='--', alpha=0.5)
plt.yscale('log') # Log scale perché i costi di fallimento possono essere enormi

plt.tight_layout()
#plt.savefig(f"comparison_summary_{ROBOT_TYPE}.png")
plt.show()

# 3. MSE (Errore di tracking)
plt.figure(figsize=(6, 4))
mse_data = [errors_mse['A'], errors_mse['B'], errors_mse['C']]
plt.boxplot(mse_data, labels=keys, patch_artist=True,
            boxprops=dict(facecolor="lightgreen"))
plt.title("Mean Square Error (Tracking Accuracy)")
plt.ylabel("MSE (Log Scale)")
plt.yscale('log')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4. TEMPI COMPUTAZIONALI
plt.figure(figsize=(6, 5))
data_times = [times['A'], times['B'], times['C']]
plt.boxplot(data_times, labels=keys, patch_artist=True, 
            boxprops=dict(facecolor="salmon"))
plt.ylabel('Simulation Time [s]')
plt.title('Computational Efficiency')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()