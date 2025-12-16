import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
import os
import shutil

# --- IMPORT MODELLI ---
# Assicuriamoci che i file dei modelli siano nella stessa cartella o nel python path
from pendolum_model import PendulumModel
from doublependolum_model import DoublePendulumModel
from neural_network import NeuralNetwork

def run_mpc_simulation(robot_type, horizon, use_network, x_init, sim_steps=100, dt=0.01):
    """
    Esegue una singola simulazione MPC.
    """
    
    # 1. SETUP ROBOT E NOMI FILE
    if robot_type == "single":
        robot = PendulumModel()
        q_des = np.array([np.pi])
        model_filename = "learned_value_pendulum.pth"
    elif robot_type == "double":
        robot = DoublePendulumModel()
        q_des = np.array([np.pi, 0.0])
        model_filename = "learned_value_double_pendulum.pth"
    else:
        raise ValueError("robot_type deve essere 'single' o 'double'")

    nq = robot.nq
    nx = robot.nx
    
    # Pesi (IDENTICI al tuo test_mpc.py funzionante)
    W_P = 1.0
    W_V = 1e-3
    W_A = 1e-4
    W_FINAL = 1.0 
    SUCCESS_THRESHOLD = 0.15 

    # --- 2. GESTIONE PERCORSI E RETE ---
    value_func = None
    
    if use_network:
        # Costruzione percorso assoluto (FIX per l'errore "non trovato")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_filename)
        
        # Pulizia preventiva build vecchie
        if os.path.exists("learned_value_function"):
            try: shutil.rmtree("learned_value_function")
            except: pass
            
        if not os.path.exists(model_path):
            print(f"WARN: Modello {model_path} non trovato! Rete disabilitata.")
            use_network = False
        else:
            # Caricamento
            checkpoint = torch.load(model_path, weights_only=False)
            input_size = checkpoint['input_size']
            hidden_size = checkpoint['hidden_size']
            ub = checkpoint['ub']
            
            # Statistiche (Reshape fondamentale per CasADi)
            mean_X = checkpoint['mean_X'].numpy().reshape(-1, 1)
            std_X = checkpoint['std_X'].numpy().reshape(-1, 1)
            mean_Y = checkpoint['mean_Y'].numpy().reshape(1, 1)
            std_Y = checkpoint['std_Y'].numpy().reshape(1, 1)
            
            # Rete
            net = NeuralNetwork(input_size, hidden_size, 1, ub=ub)
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            
            # L4Casadi Setup
            x_sym = cs.MX.sym('x', nx)
            x_norm = (x_sym - cs.DM(mean_X)) / cs.DM(std_X)
            
            # Nota: name deve essere univoco per evitare conflitti nelle compilazioni
            l4c_model = l4c.L4CasADi(net, name=f"learned_val_{robot_type}")
            
            y_norm = l4c_model(x_norm.T)
            raw_J = y_norm * cs.DM(std_Y) + cs.DM(mean_Y)
            J_pred = cs.fmax(0, raw_J) # Protezione positività
            
            value_func = cs.Function('value_func', [x_sym], [J_pred])

    # --- 3. COSTRUZIONE SOLVER (Opti) ---
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)
    param_q_des = opti.parameter(nq)
    
    X = [opti.variable(nx) for _ in range(horizon+1)]
    U = [opti.variable(nq) for _ in range(horizon)]
    
    f_dyn, inv_dyn = robot.get_dynamics_functions()
    
    cost_expr = 0
    opti.subject_to(X[0] == param_x_init)
    
    for k in range(horizon):
        # Running Cost
        err_pos = X[k][:nq] - param_q_des
        cost_expr += (W_P * cs.mtimes(err_pos.T, err_pos)) * dt
        vel = X[k][nq:]
        cost_expr += (W_V * cs.mtimes(vel.T, vel)) * dt
        cost_expr += (W_A * cs.mtimes(U[k].T, U[k])) * dt
        
        # Dinamica
        opti.subject_to(X[k+1] == X[k] + dt * f_dyn(X[k], U[k]))
        
        # Vincoli (Opzionali, come nel tuo test_mpc)
        # opti.subject_to(opti.bounded(-100.0, U[k], 100.0))
    
    # Terminal Cost (SOLO SE RETE ATTIVA)
    if use_network and value_func is not None:
        cost_expr += W_FINAL * value_func(X[-1])
        
    opti.minimize(cost_expr)
    
    # Opzioni Solver (Coerenti con test_mpc)
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1e-4,
        "ipopt.max_iter": 500,
        "print_time": 0,
        "expand": False,
        "ipopt.hessian_approximation": "limited-memory"
    }
    opti.solver("ipopt", opts)
    
    # --- 4. SIMULAZIONE ---
    current_x = x_init.copy()
    
    accumulated_cost = 0.0
    squared_error_sum = 0.0
    
    last_X_sol = None
    last_U_sol = None
    
    solver_failed = False

    x_history = [current_x]
    u_history = []
    
    for t in range(sim_steps):
        opti.set_value(param_x_init, current_x)
        opti.set_value(param_q_des, q_des)
        
        # Warm Start
        if last_X_sol is not None:
            for k in range(horizon):
                opti.set_initial(X[k], last_X_sol[k+1])
                opti.set_initial(U[k], last_U_sol[k+1] if k < horizon-1 else last_U_sol[-1])
            opti.set_initial(X[horizon], last_X_sol[horizon])
            
        try:
            sol = opti.solve()
            u_opt = sol.value(U[0])
            u_opt = np.atleast_1d(u_opt)
            
            # Clipping reale
            u_opt = np.clip(u_opt, -100.0, 100.0)
            
            last_X_sol = [sol.value(x) for x in X]
            last_U_sol = [sol.value(u) for u in U]
            
            # Calcolo metriche per report
            err_pos_val = current_x[:nq] - q_des
            vel_val = current_x[nq:]
            
            # Costo quadratico standard per confronto equo tra metodi
            # (Anche se il metodo neurale ottimizza J_NN, misuriamo la performance fisica reale)
            step_cost = (np.dot(err_pos_val, err_pos_val) + 
                         1e-3 * np.dot(vel_val, vel_val) + 
                         1e-4 * np.dot(u_opt, u_opt)) * dt
            accumulated_cost += step_cost
            
            # MSE
            squared_error_sum += np.sum(err_pos_val**2)
            
            # Step Dinamica
            f_val = robot.get_dynamics_functions()[0](current_x, u_opt)
            current_x = current_x + dt * np.array(f_val).flatten()

            x_history.append(current_x)
            u_history.append(u_opt)
            
        except RuntimeError:
            solver_failed = True
            # Recovery: input nullo
            u_null = np.zeros(nq)
            f_val = robot.get_dynamics_functions()[0](current_x, u_null)
            current_x = current_x + dt * np.array(f_val).flatten()
            accumulated_cost += 10.0 # Penalità
            last_X_sol = None
    
    # --- 5. VALUTAZIONE FINALE ---
    final_pos_error = np.linalg.norm(current_x[:nq] - q_des)
    
    # Successo se vicino al target e non esploso
    success = (final_pos_error < SUCCESS_THRESHOLD) and (not solver_failed)
    
    if np.any(np.isnan(current_x)) or np.any(np.abs(current_x) > 1e3):
        success = False
        accumulated_cost += 1e5
    
    mse = squared_error_sum / sim_steps
    
    return accumulated_cost, mse, final_pos_error, success, np.array(x_history), np.array(u_history)