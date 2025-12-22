import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm #pip install tqdm
import os

try:
    from final_project_.ocp.random_generator import generate_random_initial_states
    from final_project_.models.pendulum_model import PendulumModel
    from final_project_.models.doublependulum_model import DoublePendulumModel
    from final_project_.plot.plot import PLOTS
except ImportError:
    print("ERRORE: file non trovati. Assicurati di eseguire dalla cartella corretta.")
    exit()

# --- CONFIGURAZIONE ---
ROBOT_TYPE = "single"  # "single" o "double"
N_SAMPLES = 100000        # Anche 10000 o 100000 per il dataset col multicore
DO_PLOTS = True

# --- FUNZIONE WORKER (Esegue 1 simulazione isolata) ---
def solve_single_ocp(x_current):
    # Re-istanziamento classi necessario per ogni processo
    if ROBOT_TYPE == "single": robot = PendulumModel()
    else: robot = DoublePendulumModel()
    
    nq = robot.nq
    nx = robot.nx
    dt = 0.01     
    N = 100 
    
    opti = cs.Opti()
    param_x_init = opti.parameter(nx) 
    param_q_des = opti.parameter(nq)
    
    if nx==2: q_des = np.array([np.pi])
    else: q_des = np.array([np.pi, 0.0])

    f, inv_dyn = robot.get_dynamics_functions()
    
    # Variabili
    X, U = [], []
    for k in range(N+1): X += [opti.variable(nx)]
    for k in range(N): U += [opti.variable(nq)]

    cost = 0
    w_p, w_v, w_a = 1, 1e-3, 1e-4

    for k in range(N):     
        pos_err = X[k][:nq] - param_q_des
        cost += w_p * pos_err.T @ pos_err*dt
        vel = X[k][nq:]
        cost += w_v * vel.T @ vel*dt
        cost += w_a * U[k].T @ U[k]*dt
        opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    opti.subject_to(X[0] == param_x_init)
    opti.minimize(cost)
    
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,    # print information about execution time
        "detect_simple_bounds": True
    }
    opti.solver("ipopt", opts)
    
    opti.set_value(param_q_des, q_des)
    opti.set_value(param_x_init, x_current)

    try:
        sol = opti.solve()
        J_opt = sol.value(cost)
        # Ritorno solo i dati numerici (input, output, flag)
        return (x_current, J_opt, True)
    except:
        return (None, None, False)

# --- MAIN BLOCK ---
if __name__ == "__main__":
    time_start = clock()
    
    print(f"--- GENERAZIONE DATASET ({ROBOT_TYPE}) ---")
    print(f"Samples: {N_SAMPLES} | Cores: {multiprocessing.cpu_count()}")

    # 1. Generazione condizioni iniziali
    if ROBOT_TYPE == "single": master_robot = PendulumModel()
    else: master_robot = DoublePendulumModel()
    initial_conditions = generate_random_initial_states(master_robot, N_SAMPLES)

    # 2. Multiprocessing
    dataset_inputs = []
    dataset_labels = []
    success_count = 0

    # Pool gestisce i processi
    with Pool() as pool:
        # pool.imap restituisce i risultati uno alla volta, tqdm li conta
        results = list(tqdm(pool.imap(solve_single_ocp, initial_conditions), total=N_SAMPLES))

    # Raccolta risultati
    for x_in, J_out, success in results:
        if success:
            dataset_inputs.append(x_in)
            dataset_labels.append(J_out)
            success_count += 1

    # 3. Salvataggio
    print(f"\nFinito! Raccolti {len(dataset_inputs)} campioni validi su {N_SAMPLES}.")
    print(f"Tempo calcolo: {clock()-time_start:.2f}s")
    print(f"Value function media: {np.mean(dataset_labels) if len(dataset_labels)>0 else 0}")

    data_x = np.array(dataset_inputs) 
    data_y = np.array(dataset_labels).reshape(-1,1)

    if ROBOT_TYPE == "single": filename = "dataset_pendulum.npz"
    else: filename = "dataset_doublependulum.npz"
    np.savez(filename, inputs=data_x, targets=data_y)
    print(f"Dati salvati in '{filename}'.")

    # --- 4. GESTIONE PLOTS ---
    if DO_PLOTS and len(dataset_inputs) > 0:
        print("\n--- ESECUZIONE SINGOLA PER IL PLOT ---")
        print("Ricalcolo l'ultimo caso per generare gli oggetti grafici...")
        
        # --- Copia della definizione del problema ---
        nq, nx = master_robot.nq, master_robot.nx
        dt, N = 0.01, 100
        opti = cs.Opti()
        param_x_init = opti.parameter(nx) 
        param_q_des = opti.parameter(nq)
        
        if nx==2: q_des = np.array([np.pi])
        else: q_des = np.array([np.pi, 0.0])
        
        f, inv_dyn = master_robot.get_dynamics_functions()
        
        # Variabili per il plot
        X, U = [], []
        for k in range(N+1): X += [opti.variable(nx)]
        for k in range(N): U += [opti.variable(nq)]
        
        cost = 0
        w_p, w_v, w_a = 1, 1e-3, 1e-4
        for k in range(N):     
            pos_err = X[k][:nq] - param_q_des
            cost += w_p * pos_err.T @ pos_err*dt
            vel = X[k][nq:]
            cost += w_v * vel.T @ vel*dt
            cost += w_a * U[k].T @ U[k]*dt
            opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))
        
        opti.subject_to(X[0] == param_x_init)
        opti.minimize(cost)
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1, "detect_simple_bounds": True}
        opti.solver("ipopt", opts)
        
        # Risolvo l'ultimo caso valido per avere "sol" e "last_sol"
        last_x_init = dataset_inputs[-1]
        opti.set_value(param_q_des, q_des)
        opti.set_value(param_x_init, last_x_init)
        
        try:
            last_sol = opti.solve()
            
            # Parametri extra per il plot
            lbx = master_robot.lowerPositionLimit.tolist() + (-master_robot.velocityLimit).tolist()
            ubx = master_robot.upperPositionLimit.tolist() + master_robot.velocityLimit.tolist()
            v_max=(-master_robot.velocityLimit).tolist()
            v_min=master_robot.velocityLimit.tolist()
            tau_min = (-master_robot.effortLimit).tolist()
            tau_max = master_robot.effortLimit.tolist()
            
            # CHIAMATA ALLA FUNZIONE ORIGINALE
            print("Avvio plotting...")
            PLOTS(DO_PLOTS, success_count, last_sol, N, dt, X, U, nq, nx, inv_dyn, q_des, tau_max, tau_min, v_max, v_min, master_robot, initial_conditions)
            
        except Exception as e:
            print(f"Impossibile generare il plot finale: {e}")