import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import sys
from neural_network import NeuralNetwork
from pendulum_system import solve_ocp, discrete_dynamics

# Configurazione
sys.setrecursionlimit(2000)
np.set_printoptions(precision=4, suppress=True)

def run_closed_loop_test():
    print("--- Caricamento Rete Neurale (Double Pendulum) ---")
    
    # Assicurati che hidden_size sia 128 (o quello usato nel training)
    net = NeuralNetwork(input_size=4, hidden_size=128, output_size=1, ub=10000.0)
    # Carichiamo i pesi ignorando eventuali warning di pickle
    try:
        net.create_casadi_function("double_pendulum_value", "./", input_size=4, load_weights=True)
    except Exception as e:
        print(f"Warning nel caricamento (ignorabile se funziona): {e}")
        
    learned_value_func = net.nn_func

    # --- Parametri ---
    T_SIM = 100
    M_HORIZON = 30 
    
    x_current = np.array([0.0, 0.0, 0.0, 0.0]) # Start Down

    print(f"--- Inizio Simulazione MPC Closed-Loop (T={T_SIM}, M={M_HORIZON}) ---")

    history_x = [x_current]
    
    # Initial Guess Rumoroso (Trucco anti-blocco Step 0)
    last_X_pred = np.zeros((4, M_HORIZON + 1))
    last_U_pred = np.random.uniform(-0.5, 0.5, (2, M_HORIZON)) 
    
    # --- LOOP DI CONTROLLO ---
    for t in range(T_SIM):
        c, X_pred, U_pred, success = solve_ocp(
            x_current, 
            M_HORIZON, 
            terminal_cost_func=learned_value_func,
            prev_sol_X=last_X_pred,
            prev_sol_U=last_U_pred
        )
        
        if not success:
            if X_pred is None:
                print(f"CRITICO: Solver fallito al passo {t}.")
                break
            else:
                # Se parziale, andiamo avanti lo stesso (spesso basta per recuperare)
                pass 
            
        last_X_pred = X_pred
        last_U_pred = U_pred
        
        u_opt = U_pred[:, 0]
        x_next = discrete_dynamics(x_current, u_opt).full().flatten()
        
        history_x.append(x_next)
        x_current = x_next
        
        if t % 10 == 0:
            print(f"Step {t}/{T_SIM} | q1: {x_current[0]:.3f}, q2: {x_current[1]:.3f}")

    X_sim = np.array(history_x).T

    # --- BENCHMARK ---
    print("Calcolo Benchmark Ottimo...")
    _, X_bench, _, _ = solve_ocp(np.array([0.0, 0.0, 0.0, 0.0]), T_SIM, terminal_cost_func=None)
    
    # --- OUTPUT DATI PER ANALISI ---
    print("\n" + "="*80)
    print(">>> COPIA DA QUI IN GIÙ PER L'ANALISI <<<")
    print("Step,q1_MPC,q2_MPC,q1_Bench,q2_Bench")
    
    # Stampiamo ogni singolo step per avere i dati completi
    steps = X_sim.shape[1]
    for k in range(steps):
        # Gestione lunghezza benchmark (potrebbe variare di 1)
        if X_bench is not None and k < X_bench.shape[1]:
            q1_b, q2_b = X_bench[0, k], X_bench[1, k]
        else:
            q1_b, q2_b = 0.0, 0.0
            
        print(f"{k},{X_sim[0, k]:.4f},{X_sim[1, k]:.4f},{q1_b:.4f},{q2_b:.4f}")
        
    print(">>> FINE DATI <<<")
    print("="*80)

    # --- PLOT ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t_ax = np.arange(X_sim.shape[1])
    
    ax1.plot(t_ax, X_sim[0, :], 'g-', linewidth=2, label='Ours (MPC+NN)')
    if X_bench is not None:
        ax1.plot(X_bench[0, :], 'b--', alpha=0.5, label='Benchmark')
    ax1.axhline(np.pi, color='k', linestyle='-.', label='Target (3.14)')
    ax1.set_ylabel('q1 (rad)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t_ax, X_sim[1, :], 'g-', linewidth=2, label='Ours (MPC+NN)')
    if X_bench is not None:
        ax2.plot(X_bench[1, :], 'b--', alpha=0.5, label='Benchmark')
    ax2.axhline(0.0, color='k', linestyle='-.', label='Target (0.0)')
    ax2.set_ylabel('q2 (rad)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('final_simulation.png')
    print("Grafico salvato in 'final_simulation.png'")

if __name__ == "__main__":
    run_closed_loop_test()