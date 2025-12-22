import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from tqdm import tqdm

from final_project_.mpc.mpc_funzione import run_mpc_simulation
from final_project_.models.pendulum_model import PendulumModel
from final_project_.models.doublependulum_model import DoublePendulumModel
from final_project_.ocp.random_generator import generate_random_initial_states

# CONFIG
ROBOT_TYPE = "single" 
N_TRIALS = 100
M_VALUES = [20, 15, 10, 7, 6, 5]
DT = 0.01
SIM_STEPS = 300  

# --- FUNZIONE WORKER ---
def run_single_trial(args):
    # args è una tupla (x0, M, robot_type) per passare più argomenti a map
    x0, hor_m, r_type = args
    cost, _, _, success, _, _ = run_mpc_simulation(
        robot_type=r_type,
        horizon=hor_m,
        use_network=True,
        x_init=x0,
        sim_steps=SIM_STEPS,
        dt=DT
    )
    return success

# --- MAIN BLOCK ---
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"   STATISTICA ORIZZONTE (M) - PARALLEL STRESS TEST")
    print(f"   Robot: {ROBOT_TYPE} | Trials: {N_TRIALS}")
    print(f"{'='*60}")

    if ROBOT_TYPE == "single": temp_robot = PendulumModel()
    elif ROBOT_TYPE == "double": temp_robot = DoublePendulumModel()

    initial_states = generate_random_initial_states(temp_robot, n_samples=N_TRIALS)

    success_counts = {m: 0 for m in M_VALUES}
    avg_costs = {m: [] for m in M_VALUES} # Non usato nel plot ma mantenuto per struttura

    total_start = time.time()

    # Ciclo su M (sequenziale)
    for M in M_VALUES:
        print(f"\n>>> Testando Horizon M = {M}...", end="")
        
        # Preparo lista argomenti per il worker
        worker_args = [(x0, M, ROBOT_TYPE) for x0 in initial_states]
        
        # Esecuzione parallela dei trials
        with Pool() as pool:
            results = list(tqdm(pool.imap(run_single_trial, worker_args), total=N_TRIALS, desc=f"Progress M={M}"))
        
        # results è una lista di True/False
        current_successes = sum(results)
        success_counts[M] = current_successes
        
        rate = (current_successes / N_TRIALS) * 100
        print(f" -> Success Rate: {rate:.1f}%")

    total_end = time.time()
    print(f"\nTest completato in {(total_end - total_start):.1f} secondi.")

    # --- 3. RISULTATI E GRAFICI ---

    rates = [(success_counts[m] / N_TRIALS) * 100 for m in M_VALUES]
    labels = [str(m) for m in M_VALUES]

    # Stampa Tabella
    print("\n" + "="*40)
    print("   RISULTATI FINALI")
    print("="*40)
    print(f"{'Horizon (M)':<15} | {'Success Rate':<15} | {'Failures'}")
    print("-" * 45)
    for m, rate in zip(M_VALUES, rates):
        failures = N_TRIALS - success_counts[m]
        print(f"{m:<15} | {rate:.1f}%          | {failures}")
    print("="*40)

    # Grafico a Barre
    plt.figure(figsize=(10, 6))

    colors = ['green' if r >= 90 else 'orange' if r >= 70 else 'red' for r in rates]
    bars = plt.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')

    plt.title(f"Neural-MPC Reliability due to horizont ({ROBOT_TYPE})")
    plt.xlabel("Horizon Length (M)")
    plt.ylabel("Success Rate [%]")
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontweight='bold')

    plt.axhline(90, color='yellow', linestyle='--', label='Soglia Sicurezza (90%)')
    plt.legend()

    plt.tight_layout()
    plt.show()