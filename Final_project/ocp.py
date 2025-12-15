import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from time import time as clock

# --- IMPORTO MODELLI ---
from pendolum_model import PendulumModel
from optimal_control.casadi_adam.final_project_.prof.pendulum import Pendulum
from doublependolum_model import DoublePendulumModel

# --- IMPORTO GENERATORE RANDOMICO PER STATI INIZIALI ---
from random_generator import generate_random_initial_states

# --- IMPORTO PLOT ---
from plot import PLOTS

time_start = clock()
print("Load robot model")
robot = PendulumModel()
#robot = DoublePendulumModel() 
nq = robot.nq
nx = robot.nx

# Parametri Simulazione
DO_PLOTS = True
n_samples=1000
dt = 0.01     
N = 100     # Orizzonte lungo N

# Stato iniziale e target
q0 = np.zeros(robot.nq)     # Parte da giù (0)
dq0= np.zeros(robot.nq)
q_des= np.zeros(robot.nq)
if nx==2: q_des = np.array([np.pi])    # Target: eq. instabile (in alto)
else: q_des = np.array([np.pi, 0.0])     # Target: eq. instabile (in alto)

w_p = 1         # position weight
w_v = 1e-3      # velocity weight
w_a = 1e-4      # acceleration weight
w_final = 0     # Project A fase 1: NON usare costo terminale

print("Create optimization parameters")
opti = cs.Opti()
param_x_init = opti.parameter(nx)
param_q_des = opti.parameter(nq)
cost = 0

# --- DINAMICA ---
# già fatta nella funzione
f, inv_dyn = robot.get_dynamics_functions()

# pre-compute state and torque bounds
lbx = robot.lowerPositionLimit.tolist() + (-robot.velocityLimit).tolist()
ubx = robot.upperPositionLimit.tolist() + robot.velocityLimit.tolist()
tau_min = (-robot.effortLimit).tolist()
tau_max = robot.effortLimit.tolist()

# create all the decision variables
X, U = [], []
for k in range(N+1): 
    X += [opti.variable(nx)]
    opti.subject_to( opti.bounded(lbx, X[-1], ubx) )
for k in range(N): 
    U += [opti.variable(nq)] # Qui U è l'accelerazione (ddq)


for k in range(N):     
    # Cost function: (q - q_des)^2 + w_v * dq^2 + w_a * u^2
        
    # Errore posizione: X[k][:nq] è q
    pos_err = X[k][:nq] - param_q_des
    cost += w_p * pos_err.T @ pos_err
        
    # Penalità velocità: X[k][nq:] è dq
    vel = X[k][nq:]
    cost += w_v * vel.T @ vel
        
    # Penalità input (accelerazione)
    cost += w_a * U[k].T @ U[k]

    # Dinamica: Collocation (Eulero esplicito)
    # x_next = x + dt * f(x, u)
    opti.subject_to(X[k+1] == X[k] + dt * f(X[k], U[k]))

    # Torque constraints (tramite dinamica inversa)
    # inv_dyn: coppia necessaria per fare quell'accelerazione U[k]
    tau_k = inv_dyn(X[k], U[k])
    #opti.subject_to( opti.bounded(tau_min, tau_k, tau_max))

    # Costo finale (Non serve per la parte iniziale)
    if w_final > 0:
        cost += w_final * (X[-1][:nq] - param_q_des).T @ (X[-1][:nq] - param_q_des)
        cost += w_final * X[-1][nq:].T @ X[-1][nq:]

# Vincolo Iniziale (Legato al Parametro!)
opti.subject_to(X[0] == param_x_init)

opti.minimize(cost)

print("Create the optimization problem")
opts = {
    "ipopt.print_level": 0,
    "ipopt.tol": 1,
    "ipopt.constr_viol_tol": 1e-6,
    "ipopt.compl_inf_tol": 1e-6,
    "print_time": 0,    # print information about execution time
    "detect_simple_bounds": True
}
opti.solver("ipopt", opts)

print("Start solving the optimization problem")
# Impostiamo il target fisso (Swing Up)
#q_des_val = np.array([np.pi]) 
opti.set_value(param_q_des, q_des)

# Initial condition
print("Add initial conditions")
initial_conditions = generate_random_initial_states(robot, n_samples)

# Liste vuote per salvare i risultati
dataset_inputs = []
dataset_labels = []
J_opt=0
success_count=0

for i in range(n_samples):

    x_current = initial_conditions[i]
    opti.set_value(param_x_init, x_current)

    try:
        # Risolvi
        sol = opti.solve()
        print("SOLVED SUCCESS!")
        # Estrazione risultati
        x_sol = np.array([sol.value(X[k]) for k in range(N+1)]).T
        u_sol = np.array([sol.value(U[k]) for k in range(N)]).T # accelerazioni

        if u_sol.ndim == 1: u_sol = u_sol.reshape(nq, -1)
        if x_sol.ndim == 1: x_sol = x_sol.reshape(nx, -1)
        # ------------------------------------

        q_sol = x_sol[:nq,:]
        dq_sol = x_sol[nq:,:]
        
        # Ricalcolo coppie per il plot
        tau_vals = np.zeros((nq, N))
        for i in range(N):
            val_tau = inv_dyn(x_sol[:,i], u_sol[:,i])
            tau_vals[:,i] = val_tau.toarray().squeeze()
            
        print(f"Optimal Cost J: {sol.value(cost)}")
        J_opt=sol.value(cost)

        dataset_inputs.append(x_current) # Input
        dataset_labels.append(J_opt)     # Output

        success_count += 1

        # ultima soluzione per plot
        last_sol=sol

    except:
        print("OPTIMIZATION FAILED :(")
        # In caso di fallimento, prendiamo i valori di debug per vedere dove si è bloccato
        x_sol = np.array([opti.debug.value(X[k]) for k in range(N+1)]).T
        q_sol = x_sol[:nq,:]
        dq_sol = x_sol[nq:,:]
        tau_vals = np.zeros((nq, N)) # Vuoto
        pass

    print(f"Total time: {clock()-time_start:.2f}s")
    print(f"Success Rate: {success_count}/{n_samples}")


# --- FASE 3: SALVATAGGIO ---

print(f"Finito! Raccolti {len(dataset_inputs)} campioni validi su {n_samples}.")

# Converti le liste in array Numpy
data_x = np.array(dataset_inputs) 
data_y = np.array(dataset_labels).reshape(-1,1)

# Salva in un file compresso .npz
if nx==2: filename = "dataset_pendulum.npz"
else: filename = "dataset_doublependulum.npz"
np.savez(filename, inputs=data_x, targets=data_y)
print(f"Dati salvati in '{filename}'. Pronto per il Training!")

# --- FASE 4: PLOTS ---

PLOTS(DO_PLOTS, success_count, last_sol, N, dt, X, U, nq, nx, inv_dyn, q_des, tau_max, tau_min, robot, initial_conditions)