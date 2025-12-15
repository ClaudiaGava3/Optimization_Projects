import torch
import numpy as np
import casadi as cs
import l4casadi as l4c
import matplotlib.pyplot as plt
import os
import shutil

# --- IMPORT UTILITIES ---
from pendolum_model import PendulumModel
from doublependolum_model import DoublePendulumModel
from neural_network import NeuralNetwork
from random_generator import generate_random_initial_states
from plot import animate_pendulum, animate_double_pendulum

# --- CONFIGURAZIONE ---
#MODEL_FILE = "learned_value_pendulum.pth"       
MODEL_FILE = "learned_value_double_pendulum.pth" 

DT = 0.01
M = 20      #per pendolo semplice arriva bene anche a 10, per quello doppio già a 15 da più errore 
SIM_TIME = 3.0  
N_SIM = int(SIM_TIME / DT)

# Pesi Costo 
W_P = 1.0     
W_V = 1e-3      
W_A = 1e-4
W_FINAL = 1      

# Soglia di successo (in radianti)
# Se l'errore finale è minore di questo valore, consideriamo la prova superata.
SUCCESS_THRESHOLD = 0.15

# --- PULIZIA PREVENTIVA ---
# A volte l4casadi lascia residui di compilazione corrotti.
# Se esiste la cartella di build locale, la rimuoviamo per ricompilare pulito.
if os.path.exists("learned_value_function"):
    try:
        shutil.rmtree("learned_value_function")
    except:
        pass

# --- 1. CARICAMENTO MODELLO ---
print(f"--- Caricamento modello da {MODEL_FILE} ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, MODEL_FILE)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"File non trovato: {model_path}")

checkpoint = torch.load(model_path, weights_only=False)

input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
ub = checkpoint['ub']
robot_name = checkpoint['robot_name']

mean_X = checkpoint['mean_X'].numpy()
std_X = checkpoint['std_X'].numpy()
mean_Y = checkpoint['mean_Y'].numpy()
std_Y = checkpoint['std_Y'].numpy()

print(f"Robot rilevato: {robot_name}")
if robot_name == "pendulum":
    robot = PendulumModel()
    q_des = np.array([np.pi]) 
elif robot_name == "double_pendulum":
    robot = DoublePendulumModel()
    q_des = np.array([np.pi, 0.0]) 
else:
    raise ValueError("Robot sconosciuto")

nq = robot.nq
nx = robot.nx

net = NeuralNetwork(input_size, hidden_size, 1, ub=ub)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# --- 2. INTEGRAZIONE L4CASADI ---
print("--- Creazione Funzione L4Casadi ---")

x_sym = cs.MX.sym('x', nx)

# Normalizzazione
x_norm = (x_sym - cs.DM(mean_X)) / cs.DM(std_X)

# Inizializzazione L4Casadi
# Nota: La rete neurale PyTorch si aspetta [Batch, Dim].
# Casadi fornisce [Dim, 1]. Dobbiamo trasporre x_norm per avere [1, Dim].
l4c_model = l4c.L4CasADi(net, name="learned_value_function")

# Passaggio nella rete con TRASPOSIZIONE (.T)
y_norm = l4c_model(x_norm.T)

# Denormalizzazione
raw_J = y_norm * cs.DM(std_Y) + cs.DM(mean_Y)
J_pred = cs.fmax(0, raw_J)

value_func = cs.Function('value_func', [x_sym], [J_pred])


# --- 3. COSTRUZIONE SOLVER MPC ---
print(f"--- Costruzione MPC (M={M}) ---")
opti = cs.Opti()

param_x_init = opti.parameter(nx)
param_q_des = opti.parameter(nq)

X = [opti.variable(nx) for _ in range(M+1)]
U = [opti.variable(nq) for _ in range(M)]

f_dyn, inv_dyn = robot.get_dynamics_functions()

# pre-compute state and torque bounds
lbx = robot.lowerPositionLimit.tolist() + (-robot.velocityLimit).tolist()
ubx = robot.upperPositionLimit.tolist() + robot.velocityLimit.tolist()
tau_min = (-robot.effortLimit).tolist()
tau_max = robot.effortLimit.tolist()

cost = 0
opti.subject_to(X[0] == param_x_init)

for k in range(M):
    # Costo Running
    err_pos = X[k][:nq] - param_q_des
    cost += (W_P * cs.mtimes(err_pos.T, err_pos)) * DT
    vel = X[k][nq:]
    cost += (W_V * cs.mtimes(vel.T, vel)) * DT
    cost += (W_A * cs.mtimes(U[k].T, U[k])) * DT
    
    # Dinamica
    opti.subject_to(X[k+1] == X[k] + DT * f_dyn(X[k], U[k]))

    # --- VINCOLI DI SICUREZZA (NECESSARI PER STABILITÀ) ---
    # Velocità massima (es. 10 rad/s è già tantissimo per un pendolo di 1m)
    #opti.subject_to(opti.bounded(-15.0, X[k+1][nq:], 15.0))
    #opti.subject_to(opti.bounded(-100.0, U[k], 100.0))

# --- COSTO TERMINALE ---
J_terminal = value_func(X[-1])
print(J_terminal)
cost += W_FINAL*J_terminal

opti.minimize(cost)

# --- OPZIONI SOLVER (CRUCIALE PER L4CASADI) ---
opts = {
    "ipopt.print_level": 0,
    "ipopt.tol": 1e-4,
    "ipopt.max_iter": 500,
    "print_time": 0,
    "expand": False,  # Disabilita espansione SX (incompatibile con L4Casadi)
    
    # *** FIX CRUCIALE PER L'ERRORE eval_h ***
    # Diciamo a IPOPT di approssimare l'Hessiano (limited-memory BFGS)
    # invece di chiederlo esatto alla rete neurale (che non ce l'ha).
    "ipopt.hessian_approximation": "limited-memory" 
}
opti.solver("ipopt", opts)


# --- 4. SIMULAZIONE ---
# Generiamo uno stato casuale
x_init = generate_random_initial_states(robot, n_samples=1)[0]

# --- DEBUG: Proviamo a forzare uno stato semplice se il random è troppo difficile ---
#x_init = np.array([0.0, 0.0]) # Pendolo giù (test swing up)

print(f"\nStato Iniziale: {x_init}")
print(f"Target: {q_des}")

x_history = [x_init]
u_history = []

current_x = x_init.copy()
last_X_sol = None
last_U_sol = None

print("\nAvvio loop di controllo...")
for t in range(N_SIM):
    opti.set_value(param_x_init, current_x)
    opti.set_value(param_q_des, q_des)
    
    if last_X_sol is not None:
        for k in range(M):
            opti.set_initial(X[k], last_X_sol[k+1])
            opti.set_initial(U[k], last_U_sol[k+1] if k < M-1 else last_U_sol[-1])
        opti.set_initial(X[M], last_X_sol[M])
    
    try:
        sol = opti.solve()
        
        u_opt = sol.value(U[0])
        u_opt = np.atleast_1d(u_opt) # Assicura che sia array
        
        last_X_sol = [sol.value(x) for x in X]
        last_U_sol = [sol.value(u) for u in U]
        
        # Simulazione
        f_val = robot.get_dynamics_functions()[0](current_x, u_opt)
        next_x = current_x + DT * np.array(f_val).flatten()
        
        x_history.append(next_x)
        u_history.append(u_opt)
        
        current_x = next_x
        
        if t % 50 == 0:
            print(f"Step {t}/{N_SIM} - Costo J: {sol.value(cost):.2f}")
            
    except RuntimeError as e:
        print(f"Errore solver al passo {t}. Stato corrente: {current_x}")
        # Se fallisce al primo step, è fatale.
        if t == 0:
            print("Fallimento critico all'avvio. Esco.")
            exit()
        break

print("Simulazione terminata.")

# --- 5. PLOT E ANIMAZIONE ---
if len(u_history) == 0:
    print("Nessun dato da plottare.")
    exit()

x_history = np.array(x_history)
u_history = np.array(u_history)
time_axis = np.arange(len(x_history)) * DT

# Reshape U per sicurezza nel plot
if u_history.ndim == 1:
    u_history = u_history.reshape(-1, 1)



# --- CALCOLO METRICA DI SUCCESSO ---
# Prendiamo l'ultima configurazione dei giunti (escludiamo le velocità)
q_final = x_history[-1, :nq]
# Calcoliamo la distanza euclidea dal target
final_error = np.linalg.norm(q_final - q_des)
success = final_error < SUCCESS_THRESHOLD

print("\n" + "="*30)
print(f"   RISULTATI FINALI")
print("="*30)
print(f"Target:           {q_des}")
print(f"Posizione Finale: {q_final}")
print(f"Errore (Norma):   {final_error:.4f} rad")
print(f"Soglia Successo:  {SUCCESS_THRESHOLD} rad")
print("-" * 30)
if success:
    print(f"ESITO: SUCCESSO ✅")
else:
    print(f"ESITO: FALLITO ❌ (Errore troppo alto)")
print("="*30 + "\n")



# Plot Stati
plt.figure(figsize=(10, 6))
for j in range(nq):
    plt.plot(time_axis, x_history[:, j], label=f'q_{j+1}')
    plt.axhline(q_des[j], color='gray', linestyle='--', alpha=0.5)
plt.title(f"Posizioni - Neural MPC (M={M})")
plt.legend(); plt.grid(True); plt.show()

# Plot Controlli
plt.figure(figsize=(10, 4))
valid_len = len(u_history)
for j in range(nq):
    plt.plot(time_axis[:valid_len], u_history[:, j], label=f'u_{j+1}')
plt.title("Controlli")
plt.grid(True); plt.legend(); plt.show()

# Animazione
print("Generazione Animazione...")
q_anim = x_history[:, :nq].T 

if nq == 1:
    animate_pendulum(time_axis, q_anim[0,:], length=robot.l, dt=DT)
elif nq == 2:
    animate_double_pendulum(time_axis, q_anim, l1=robot.l1, l2=robot.l2, dt=DT)