import numpy as np
import matplotlib.pyplot as plt
import time

# --- IMPORTAZIONI DAI TUOI FILE ESISTENTI ---
from optimal_control.casadi_adam.final_project_.mpc.mpc_funzione import run_mpc_simulation
from optimal_control.casadi_adam.final_project_.models.pendulum_model import PendulumModel
from optimal_control.casadi_adam.final_project_.models.doublependulum_model import DoublePendulumModel
from optimal_control.casadi_adam.final_project_.ocp.random_generator import generate_random_initial_states

# robot da testare: "single" o "double"
ROBOT_TYPE = "double" 

# Numero di prove casuali (più alto è, più la statistica è affidabile)
N_TRIALS = 100

M_VALUES = [20, 15, 10, 7, 6, 5]

# Parametri Simulazione
DT = 0.01
SIM_STEPS = 300  # 3 secondi

# --- 1. GENERAZIONE DATASET DI TEST (UGUALE PER TUTTI GLI M) ---
print(f"\n{'='*60}")
print(f"   STATISTICA ORIZZONTE (M) - STRESS TEST")
print(f"   Robot: {ROBOT_TYPE} | Trials: {N_TRIALS}")
print(f"{'='*60}")

print("Generazione condizioni iniziali condivise...")
if ROBOT_TYPE == "single":
    temp_robot = PendulumModel()
elif ROBOT_TYPE == "double":
    temp_robot = DoublePendulumModel()

# Generazione gli stati iniziali una volta
initial_states = generate_random_initial_states(temp_robot, n_samples=N_TRIALS)

# Struttura per salvare i risultati
success_counts = {m: 0 for m in M_VALUES}
# Salvo anche i tempi medi di calcolo
avg_costs = {m: [] for m in M_VALUES}

# --- 2. LOOP DI TEST ---
total_start = time.time()

for M in M_VALUES:
    print(f"\n>>> Testando Horizon M = {M}...")
    
    current_successes = 0
    
    for i, x0 in enumerate(initial_states):
        
        if i % 10 == 0: print(f"    Trial {i}/{N_TRIALS}...", end="\r")
        
        cost, _, _, success, _, _ = run_mpc_simulation(
            robot_type=ROBOT_TYPE,
            horizon=M,
            use_network=True,
            x_init=x0,
            sim_steps=SIM_STEPS,
            dt=DT
        )
        
        if success:
            current_successes += 1
            avg_costs[M].append(cost)
            
    success_counts[M] = current_successes
    rate = (current_successes / N_TRIALS) * 100
    print(f"    -> Completato. Success Rate: {rate:.1f}% ({current_successes}/{N_TRIALS})")

total_end = time.time()
print(f"\nTest completato in {(total_end - total_start):.1f} secondi.")

# --- 3. RISULTATI E GRAFICI ---

# Calcolo percentuali
rates = [(success_counts[m] / N_TRIALS) * 100 for m in M_VALUES]
# M convertiti in stringhe per il grafico
labels = [str(m) for m in M_VALUES]

# Stampa Tabella Finale
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

# Colore barre: Verde se > 90%, Giallo se > 70%, Rosso altrimenti
colors = ['green' if r >= 90 else 'orange' if r >= 70 else 'red' for r in rates]

bars = plt.bar(labels, rates, color=colors, alpha=0.7, edgecolor='black')

plt.title(f"Neural-MPC Reliability due to horizont ({ROBOT_TYPE})")
plt.xlabel("Horizon Length (M)")
plt.ylabel("Success Rate [%]")
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Aggiungo il valore percentuale sopra ogni barra
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold')

# Linea di soglia "accettabile" (90%)
plt.axhline(90, color='yellow', linestyle='--', label='Soglia Sicurezza (90%)')
plt.legend()

plt.tight_layout()
plt.show()