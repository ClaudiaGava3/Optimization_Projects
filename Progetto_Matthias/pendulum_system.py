import casadi as ca
import numpy as np

# --- 1. Parametri Fisici ---
m1 = 1.0; m2 = 1.0
l1 = 1.0; l2 = 1.0
g  = 9.81 

nq, nx, nu = 2, 4, 2

# --- 2. Dinamica ---
def dynamics_double_pendulum(x, u):
    q1, q2 = x[0], x[1]
    dq1, dq2 = x[2], x[3]
    
    M11 = (m1 + m2) * l1**2
    M12 = m2 * l1 * l2 * ca.cos(q1 - q2)
    M22 = m2 * l2**2
    M = ca.vertcat(ca.horzcat(M11, M12), ca.horzcat(M12, M22))
    
    C1 = -m2 * l1 * l2 * dq2**2 * ca.sin(q1 - q2)
    C2 =  m2 * l1 * l2 * dq1**2 * ca.sin(q1 - q2)
    C_terms = ca.vertcat(C1, C2)
    
    G1 = (m1 + m2) * g * l1 * ca.sin(q1)
    G2 = m2 * g * l2 * ca.sin(q2)
    G = ca.vertcat(G1, G2)
    
    ddq = ca.solve(M, u - C_terms - G)
    return ca.vertcat(dq1, dq2, ddq)

def discrete_dynamics(x, u, dt=0.02):
    k1 = dynamics_double_pendulum(x, u)
    k2 = dynamics_double_pendulum(x + dt/2 * k1, u)
    k3 = dynamics_double_pendulum(x + dt/2 * k2, u)
    k4 = dynamics_double_pendulum(x + dt * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. Cost Function ---
def cost_function(x, u):
    target_state = ca.DM([np.pi, 0.0, 0.0, 0.0]) 
    e = x - target_state
    Q = ca.diag([20.0, 20.0, 5.0, 5.0]) 
    R = ca.diag([0.1, 0.1])
    return e.T @ Q @ e + u.T @ R @ u

# --- 4. Costruttore Parametrico (Per Generazione Dati) ---
def build_parametric_ocp(horizon):
    opti = ca.Opti()
    p_x_init = opti.parameter(4)
    X = opti.variable(4, horizon + 1)
    U = opti.variable(2, horizon)
    J = 0
    opti.subject_to(X[:, 0] == p_x_init)
    
    for i in range(horizon):
        opti.subject_to(X[:, i+1] == discrete_dynamics(X[:, i], U[:, i]))
        J += cost_function(X[:, i], U[:, i])
        opti.subject_to(opti.bounded(-15.0, U[:, i], 15.0))
        
    opti.minimize(J)
    
    # Opzioni Minimali e Sicure
    opts = {
        'ipopt.print_level': 0, 
        'print_time': False,       # Importante: False (bool), non 0 (int)
        'ipopt.tol': 1e-2, 
        'ipopt.max_iter': 150
    }
    opti.solver('ipopt', opts)
    return opti, p_x_init, X, U, J

# --- 5. Risolutore Classico (Per Test MPC) ---
def solve_ocp(x_init, horizon, terminal_cost_func=None, prev_sol_X=None, prev_sol_U=None):
    opti = ca.Opti()
    X = opti.variable(4, horizon + 1)
    U = opti.variable(2, horizon)
    J = 0
    
    # Inizializzazione Manuale (Warm Start Semplificato)
    if prev_sol_X is not None:
        opti.set_initial(X[:, :-1], prev_sol_X[:, 1:])
        opti.set_initial(X[:, -1],  prev_sol_X[:, -1])
    else:
        opti.set_initial(X, ca.repmat(x_init, 1, horizon + 1))
        
    if prev_sol_U is not None:
        if prev_sol_U.shape[1] > 1:
            opti.set_initial(U[:, :-1], prev_sol_U[:, 1:])
            opti.set_initial(U[:, -1],  prev_sol_U[:, -1])
        else:
             opti.set_initial(U, prev_sol_U)
        
    opti.subject_to(X[:, 0] == x_init)
    
    for i in range(horizon):
        opti.subject_to(X[:, i+1] == discrete_dynamics(X[:, i], U[:, i]))
        J += cost_function(X[:, i], U[:, i])
        opti.subject_to(opti.bounded(-15.0, U[:, i], 15.0))
        
    if terminal_cost_func is not None:
        J += terminal_cost_func(X[:, horizon])
        
    opti.minimize(J)
    
    # --- OPZIONI SICURE ---
    # Rimosse stringhe come 'yes'/'no' che causano errori di tipo
    opts = {
        'ipopt.print_level': 5,      # Integer
        'print_time': False,         # Boolean
        'ipopt.tol': 1e-3,           # Float
        'ipopt.max_iter': 3000       # Integer
    }
    opti.solver('ipopt', opts)
    
    try:
        sol = opti.solve()
        return sol.value(J), sol.value(X), sol.value(U), True
    except Exception as e:
        # Se fallisce, ora vedremo l'errore matematico reale (es. Infeasible), non quello di tipo.
        print(f"\n[SOLVER EXCEPTION]: {e}")
        try:
            return opti.debug.value(J), opti.debug.value(X), opti.debug.value(U), False
        except:
            return 0, None, None, False