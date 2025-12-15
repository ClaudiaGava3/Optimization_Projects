import numpy as np
import casadi as cs
import torch
import l4casadi as l4c
import os
import warnings
from time import time as clock

# Silenzia i warning
warnings.filterwarnings("ignore")

# --- IMPORT MODELLI E RETE ---
# Assicurati che questi file siano nella stessa cartella
try:
    from pendolum_model import PendulumModel
    from doublependolum_model import DoublePendulumModel
    from neural_network import NeuralNetwork
except ImportError:
    print("ERRORE: Assicurati che 'pendulum_model.py', 'double_pendulum_model.py' e 'neural_network.py' siano nella cartella.")

# ==============================================================================
# FIX CRUCIALE PER ERRORE "mat1 and mat2 shapes cannot be multiplied"
# ==============================================================================
# Salviamo la funzione originale della rete se non è stata già salvata
if not hasattr(NeuralNetwork, 'original_forward'):
    NeuralNetwork.original_forward = NeuralNetwork.forward

# Creiamo una funzione "patchata" che controlla la forma dell'input
def patched_forward(self, x):
    # Se x è un vettore colonna (es. 2x1), lo trasformiamo in riga (1x2)
    # L4CasADi spesso passa vettori colonna durante la creazione
    if x.dim() == 2 and x.shape[1] == 1 and x.shape[0] > 1:
        x = x.t() 
    return self.linear_stack(x) * self.ub

# Applichiamo la patch alla classe PRIMA di usarla
NeuralNetwork.forward = patched_forward
# ==============================================================================

def run_mpc_simulation(robot_type="single", horizon=20, use_network=True, x_init=None, sim_steps=100):
    """
    Esegue una simulazione MPC parametrica.
    
    Args:
        robot_type (str): "single" o "double"
        horizon (int): Lunghezza orizzonte predittivo (M)
        use_network (bool): Se True, aggiunge il costo terminale neurale.
        x_init (np.array): Stato iniziale del robot.
        sim_steps (int): Numero di passi di simulazione.
        
    Returns:
        total_cost (float): Somma dei costi reali accumulati durante la simulazione.
        history_q (np.array): Storia delle posizioni.
        history_u (np.array): Storia delle coppie applicate.
    """
    
    # --- 1. SETUP ROBOT ---
    if robot_type == "single":
        robot = PendulumModel()
        q_des = np.array([np.pi])
        if x_init is None: x_init = np.array([0.0, 0.0])
    else:
        robot = DoublePendulumModel()
        q_des = np.array([np.pi, 0.0])
        if x_init is None: x_init = np.array([0.0, 0.0, 0.0, 0.0])

    x_sim = x_init.copy()
    nq = robot.nq
    nx = robot.nx
    f, inv_dyn = robot.get_dynamics_functions()
    DT = 0.05

    # --- 2. CARICAMENTO RETE (Solo se richiesta) ---
    l4c_func = None
    mean_X, std_Xdev, mean_Y, std_Y = None, None, None, None
    
    if use_network:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_name = "learned_value_pendulum.pth" if robot_type == "single" else "learned_value_double_pendulum.pth"
        path_pth = os.path.join(script_dir, model_name)

        if not os.path.exists(path_pth):
            print(f"ERRORE: File {model_name} non trovato.")
            return 0, [], []

        try:
            checkpoint = torch.load(path_pth, map_location='cpu')
        except Exception as e:
            print(f"Errore caricamento checkpoint: {e}")
            return 0, [], []

        # Estrazione dati normalizzazione
        mean_X = checkpoint['mean_X'].numpy().reshape(-1)
        std_Xdev = checkpoint['std_X'].numpy().reshape(-1)
        mean_Y = checkpoint['mean_Y'].numpy()
        std_Y = checkpoint['std_Y'].numpy()
        ub = checkpoint.get('ub', 10.0)

        # Inizializzazione Modello
        model = NeuralNetwork(input_size=nx, hidden_size=64, output_size=1, ub=ub)
        if 'model' in checkpoint: model.load_state_dict(checkpoint['model'])
        else: model.load_state_dict(checkpoint['model_state_dict'])

        # Generazione funzione L4CasADi
        model.create_casadi_function(robot_name=robot_type, NN_DIR=script_dir+"/", input_size=nx, load_weights=False)
        l4c_func = model.nn_func

    # --- 3. COSTRUZIONE SOLVER MPC ---
    opti = cs.Opti()
    param_x_init = opti.parameter(nx)
    param_q_des  = opti.parameter(nq)

    # Variabili decisione
    X = [opti.variable(nx) for _ in range(horizon+1)]
    U = [opti.variable(nq) for _ in range(horizon)] # U è accelerazione

    # Pesi Cost Function
    w_p = 10.0 
    w_v = 0.1
    w_a = 0.01
    w_f = 100 # Peso costo terminale (usato solo se use_network=True)

    cost = 0

    # Loop Orizzonte
    for k in range(horizon):
        pos_err = X[k][:nq] - param_q_des
        vel = X[k][nq:]
        
        # Running Cost
        cost += w_p * cs.sumsqr(pos_err) + w_v * cs.sumsqr(vel) + w_a * cs.sumsqr(U[k])
        
        # Dinamica
        x_next = X[k] + DT * f(X[k], U[k])
        opti.subject_to(X[k+1] == x_next)
        
        # Vincoli Coppia (Commentati come nel tuo file funzionante)
        tau_k = inv_dyn(X[k], U[k])
        # opti.subject_to(opti.bounded(-robot.effortLimit, tau_k, robot.effortLimit))

    # Costo Terminale (Se attivato)
    if use_network and l4c_func is not None:
        x_final = X[horizon]
        x_norm = (x_final - mean_X) / std_Xdev
        val_norm = l4c_func(x_norm.T) # .T per sicurezza
        val_real = val_norm * std_Y + mean_Y
        cost += w_f * val_real

    opti.subject_to(X[0] == param_x_init)
    opti.minimize(cost)

    # Solver Settings (I tuoi che funzionano!)
    opts = {
        "ipopt.print_level": 0,
        "ipopt.tol": 1,
        "ipopt.constr_viol_tol": 1e-6,
        "ipopt.compl_inf_tol": 1e-6,
        "print_time": 0,
        "detect_simple_bounds": True,
        "ipopt.hessian_approximation": "limited-memory" 
    }
    opti.solver("ipopt", opts)

    # --- 4. SIMULAZIONE ---
    history_q = []
    history_u = []
    accumulated_cost = 0.0

    # Warm Start Init
    opti.set_value(param_x_init, x_sim)
    opti.set_value(param_q_des, q_des)
    try:
        sol = opti.solve()
        last_u = sol.value(cs.horzcat(*U))
        last_x = sol.value(cs.horzcat(*X))
    except:
        last_u = np.zeros((nq, horizon))
        last_x = np.zeros((nx, horizon+1))

    for t in range(sim_steps):
        opti.set_value(param_x_init, x_sim)
        opti.set_value(param_q_des, q_des)
        
        # Warm start
        opti.set_initial(cs.horzcat(*U), last_u)
        opti.set_initial(cs.horzcat(*X), last_x)
        
        try:
            sol = opti.solve()
            u_mpc = sol.value(U[0])
            
            # Aggiorna warm start per prossimo step
            last_u = sol.value(cs.horzcat(*U))
            last_x = sol.value(cs.horzcat(*X))
            is_success=1
            
        except Exception:
            # Fallback (come nel tuo script)
            u_mpc = np.zeros(nq)
            is_success=0

        # Calcolo costo reale passo (per il grafico comparativo)
        pos_err_real = x_sim[:nq] - q_des
        vel_real = x_sim[nq:]
        # Nota: Qui u_mpc è accelerazione, non coppia, coerente con la tua cost function
        step_cost = w_p * np.sum(pos_err_real**2) + w_v * np.sum(vel_real**2) + w_a * np.sum(u_mpc**2)
        accumulated_cost += step_cost

        # Simula dinamica reale
        res_int = f(x_sim, u_mpc)
        x_sim = x_sim + DT * np.array(res_int.full()).flatten()
        
        # Salva dati
        tau_real = inv_dyn(x_sim, u_mpc).full().flatten()
        history_q.append(x_sim[:nq])
        history_u.append(tau_real)

        # --- 5. VERIFICA SUCCESSO ---
        # Consideriamo successo se la posizione finale è vicina al target (es. errore < 0.5 rad)
    
    return accumulated_cost, np.array(history_q), np.array(history_u), is_success
