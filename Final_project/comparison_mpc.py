import numpy as np
import matplotlib.pyplot as plt
from mpc_funzione import run_mpc_simulation

# --- IMPORTA IL GENERATORE RANDOM ---
try:
    from random_generator import generate_random_initial_states
    from pendolum_model import PendulumModel
    from doublependolum_model import DoublePendulumModel
except ImportError:
    print("ERRORE: random_generator.py o modelli non trovati.")
    exit()

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "single"  # "single" o "double"
N_TRIALS = 20          # Numero prove random (partiamo con 20 per velocità)
M_SHORT = 20           # Orizzonte Corto (quello che hai usato con successo)
N_LONG = 100           # Orizzonte Lungo (Benchmark)
SIM_STEPS = 150        # Durata simulazione (step) per dare tempo di arrivare

print(f"--- INIZIO COMPARAZIONE (Robot: {ROBOT_TYPE}) ---")
print(f"Horizon Short: {M_SHORT}, Horizon Long: {N_LONG}")
print(f"Trials: {N_TRIALS}")

# Setup robot temporaneo per generare stati
if ROBOT_TYPE == "single": temp_robot = PendulumModel()
else: temp_robot = DoublePendulumModel()

# --- GENERAZIONE STATI INIZIALI ---
# Generiamo N stati casuali validi.
# Usiamo il tuo generatore
initial_states = generate_random_initial_states(temp_robot, n_samples=N_TRIALS)

# Strutture dati per risultati
# Costi (più basso è meglio)
costs = {'A': [], 'B': [], 'C': []}
# Errori MSE rispetto al target
errors_mse = {'A': [], 'B': [], 'C': []}
# Tassi di successo
success_flags = {'A': [], 'B': [], 'C': []}

# --- LOOP DI TEST ---
for i in range(N_TRIALS):
    x0 = initial_states[i]
    print(f"\n>>> Trial {i+1}/{N_TRIALS} - Init: {x0}")
    
    # 1. CASO A: Orizzonte Corto (M), No Rete
    # Questo ci aspettiamo che fallisca o faccia fatica
    print("   Running Case A (Short, No NN)...", end="")
    cost_a, mse_a, dist_a, ok_a = run_mpc_simulation(ROBOT_TYPE, horizon=M_SHORT, use_network=False, x_init=x0, sim_steps=SIM_STEPS)
    costs['A'].append(cost_a)
    errors_mse['A'].append(mse_a)
    success_flags['A'].append(ok_a)
    print(f" Done. Success: {ok_a}, Cost: {cost_a:.1f}")

    # 2. CASO B: Orizzonte Corto (M), CON Rete -> Il tuo metodo
    print("   Running Case B (Short + NeuralNet)...", end="")
    cost_b, mse_b, dist_b, ok_b = run_mpc_simulation(ROBOT_TYPE, horizon=M_SHORT, use_network=True, x_init=x0, sim_steps=SIM_STEPS)
    costs['B'].append(cost_b)
    errors_mse['B'].append(mse_b)
    success_flags['B'].append(ok_b)
    print(f" Done. Success: {ok_b}, Cost: {cost_b:.1f}")
    
    # 3. CASO C: Orizzonte Lungo (Benchmark), No Rete
    # Questo è lento ma dovrebbe essere il migliore
    print("   Running Case C (Long Benchmark)...", end="")
    cost_c, mse_c, dist_c, ok_c = run_mpc_simulation(ROBOT_TYPE, horizon=N_LONG, use_network=False, x_init=x0, sim_steps=SIM_STEPS)
    costs['C'].append(cost_c)
    errors_mse['C'].append(mse_c)
    success_flags['C'].append(ok_c)
    print(f" Done. Success: {ok_c}, Cost: {cost_c:.1f}")

# --- ANALISI DATI ---
def get_stats(data_list):
    return np.mean(data_list), np.std(data_list)

print("\n" + "="*40)
print("   REPORT FINALE")
print("="*40)

methods = [('A', f'Short (N={M_SHORT})'), 
           ('B', f'Neural (N={M_SHORT})'), 
           ('C', f'Long (N={N_LONG})')]

# Calcolo percentuali successo
succ_rates = {}
for k, label in methods:
    succ_rate = np.sum(success_flags[k]) / N_TRIALS * 100
    succ_rates[k] = succ_rate
    
    # Calcoliamo costo medio SOLO sulle prove di successo (per non falsare con costi infiniti)
    # Oppure su tutte se vogliamo penalizzare i fallimenti
    valid_costs = [c for c, ok in zip(costs[k], success_flags[k])] # prendiamo tutto per ora
    mean_c = np.mean(costs[k])
    
    mean_mse = np.mean(errors_mse[k])
    
    print(f"Metodo {k} [{label}]:")
    print(f"  Success Rate: {succ_rate:.1f}%")
    print(f"  Avg Cost:     {mean_c:.2f}")
    print(f"  Avg MSE:      {mean_mse:.4f}")
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
plt.title(f'Affidabilità ({ROBOT_TYPE})')
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)
# Aggiungi etichette sulle barre
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.0f}%", ha='center', va='bottom')

# 2. Costo Medio (Box Plot per vedere la varianza)
plt.subplot(1, 2, 2)
data_to_plot = [costs['A'], costs['B'], costs['C']]
plt.boxplot(data_to_plot, labels=keys, patch_artist=True, 
            boxprops=dict(facecolor="lightblue"))
plt.ylabel('Total Cost (Accumulated)')
plt.title('Distribuzione dei Costi')
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
plt.title("Errore Quadratico Medio (Tracking Accuracy)")
plt.ylabel("MSE (Log Scale)")
plt.yscale('log')
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()