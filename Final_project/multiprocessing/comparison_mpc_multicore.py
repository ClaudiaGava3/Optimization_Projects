import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm 

from final_project_.mpc.mpc_funzione import run_mpc_simulation

# Import generatore e modelli
try:
    from final_project_.ocp.random_generator import generate_random_initial_states
    from final_project_.models.pendulum_model import PendulumModel
    from final_project_.models.doublependulum_model import DoublePendulumModel
except ImportError:
    print("ERRORE: random_generator.py o modelli non trovati.")
    exit()

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "single"  # "single" o "double"
N_TRIALS = 100         # Numero prove random
M_SHORT = 5            # Orizzonte Corto per single
#M_SHORT = 20          # Orizzonte Corto per double
N_LONG = 100           # Orizzonte Lungo
SIM_STEPS = 300        

# --- FUNZIONE WORKER ---
def test_single_scenario(x0):
    # CASO A
    t0 = clock()
    c_a, m_a, _, ok_a, _, _ = run_mpc_simulation(ROBOT_TYPE, M_SHORT, False, x0, SIM_STEPS)
    t_a = clock() - t0
    
    # CASO B
    t0 = clock()
    c_b, m_b, _, ok_b, _, _ = run_mpc_simulation(ROBOT_TYPE, M_SHORT, True, x0, SIM_STEPS)
    t_b = clock() - t0
    
    # CASO C
    t0 = clock()
    c_c, m_c, _, ok_c, _, _ = run_mpc_simulation(ROBOT_TYPE, N_LONG+M_SHORT, False, x0, SIM_STEPS)
    t_c = clock() - t0

    # Ritorna tuple di dati per ogni metodo
    return (c_a, m_a, ok_a, t_a), (c_b, m_b, ok_b, t_b), (c_c, m_c, ok_c, t_c)

# --- MAIN BLOCK ---
if __name__ == "__main__":
    print(f"--- INIZIO COMPARAZIONE PARALLELA (Robot: {ROBOT_TYPE}) ---")
    print(f"Horizon Short: {M_SHORT}, Horizon Long: {N_LONG} | Trials: {N_TRIALS}")

    # Generazione stati
    if ROBOT_TYPE == "single": temp_robot = PendulumModel()
    else: temp_robot = DoublePendulumModel()
    initial_states = generate_random_initial_states(temp_robot, n_samples=N_TRIALS)

    # Multiprocessing
    print(f"Avvio simulazioni su {multiprocessing.cpu_count()} core...")
    t_start_all = clock()
    
    with Pool() as pool:
        # Aggiungo la barra di progresso
        results = list(tqdm(pool.imap(test_single_scenario, initial_states), total=N_TRIALS))
    
    print(f"Simulazioni finite in {clock()-t_start_all:.2f}s. Elaborazione dati...")

    # Ricostruzione liste per i grafici
    costs = {'A': [], 'B': [], 'C': []}
    errors_mse = {'A': [], 'B': [], 'C': []}
    success_flags = {'A': [], 'B': [], 'C': []}
    times = {'A': [], 'B': [], 'C': []}

    for res_A, res_B, res_C in results:
        # A
        costs['A'].append(res_A[0]); errors_mse['A'].append(res_A[1]); success_flags['A'].append(res_A[2]); times['A'].append(res_A[3])
        # B
        costs['B'].append(res_B[0]); errors_mse['B'].append(res_B[1]); success_flags['B'].append(res_B[2]); times['B'].append(res_B[3])
        # C
        costs['C'].append(res_C[0]); errors_mse['C'].append(res_C[1]); success_flags['C'].append(res_C[2]); times['C'].append(res_C[3])

    # --- ANALISI DATI E PLOTS ---
    
    print("\n" + "="*40)
    print("   REPORT FINALE")
    print("="*40)

    methods = [('A', f'Short (M={M_SHORT})'), 
               ('B', f'Neural (M={M_SHORT})'), 
               ('C', f'Long (N+M={N_LONG+M_SHORT})')]

    succ_rates = {}
    for k, label in methods:
        succ_rate = np.sum(success_flags[k]) / N_TRIALS * 100
        succ_rates[k] = succ_rate
        
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

    # 1. Tasso di Successo
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    rates = [succ_rates[k] for k in keys]
    bars = plt.bar(keys, rates, color=['gray', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Success Rate (%)')
    plt.title(f'Reliability ({ROBOT_TYPE})')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.0f}%", ha='center', va='bottom')

    # 2. Costo Medio
    plt.subplot(1, 2, 2)
    data_to_plot = [costs['A'], costs['B'], costs['C']]
    plt.boxplot(data_to_plot, labels=keys, patch_artist=True, 
                boxprops=dict(facecolor="lightblue"))
    plt.ylabel('Total Cost (Accumulated)')
    plt.title('Cost Distribution')
    plt.grid(linestyle='--', alpha=0.5)
    plt.yscale('log') 

    plt.tight_layout()
    plt.show()

    # 3. MSE
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

    # 4. TEMPI
    plt.figure(figsize=(6, 5))
    data_times = [times['A'], times['B'], times['C']]
    plt.boxplot(data_times, labels=keys, patch_artist=True, 
                boxprops=dict(facecolor="salmon"))
    plt.ylabel('Simulation Time [s]')
    plt.title('Computational Efficiency')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()