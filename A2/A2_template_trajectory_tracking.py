#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
import time
from time import sleep

from adam.casadi.computations import KinDynComputations
from example_robot_data.robots_loader import load
import orc.optimal_control.casadi_adam.conf_ur5 as conf_ur5
from orc.utils.robot_wrapper import RobotWrapper
from orc.utils.robot_simulator import RobotSimulator
from orc.utils.viz_utils import addViewerSphere, applyViewerConfiguration


# ====================== Simple plotting utility ======================
def plot_infinity(t_init, t_final):
    r = 0.1
    t = np.linspace(t_init, t_final, 300)
    x = r * np.cos(2*np.pi*t)
    y = r * 0.5*np.sin(4*np.pi*t)
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'x')
    plt.xlim([-0.11, 0.11])
    plt.ylim([-0.06, 0.06])
    plt.title("Infinity-shaped path")
    plt.grid(True)
    plt.show()


# ====================== Robot and Dynamics Setup ======================
robot = load("ur5")
joints_name_list = [s for s in robot.model.names[1:]]
nq = len(joints_name_list)
nx = 2 * nq

#dt = 0.02
N = 500
q0 = np.zeros(nq)
dq0 = np.zeros(nq)
x_init = np.concatenate([q0, dq0])
w_max = 5
frame_name = "ee_link"
r_path = 0.2

# CasADi symbolic variables
q = cs.SX.sym('q', nq)
dq = cs.SX.sym('dq', nq)
ddq = cs.SX.sym('ddq', nq)
state = cs.vertcat(q, dq)
rhs = cs.vertcat(dq, ddq)
f = cs.Function('f', [state, ddq], [rhs])

# Inverse dynamics
kinDyn = KinDynComputations(robot.urdf, joints_name_list)
H_b = cs.SX.eye(4)
v_b = cs.SX.zeros(6)
bias_forces = kinDyn.bias_force_fun()
mass_matrix = kinDyn.mass_matrix_fun()
h = bias_forces(H_b, q, v_b, dq)[6:]
M = mass_matrix(H_b, q)[6:, 6:]
tau = M @ ddq + h
inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

# Forward kinematics
fk_fun = kinDyn.forward_kinematics_fun(frame_name)
ee_pos = fk_fun(H_b, q)[:3, 3]
fk = cs.Function('fk', [q], [ee_pos])

y = fk(q0)
c_path = np.array([y[0]-r_path, y[1], y[2]]).squeeze()

lbx = robot.model.lowerPositionLimit.tolist() + (-robot.model.velocityLimit).tolist()
ubx = robot.model.upperPositionLimit.tolist() + robot.model.velocityLimit.tolist()
tau_min = (-robot.model.effortLimit).tolist()
tau_max = robot.model.effortLimit.tolist()


# ====================== Optimization Problem ======================
def create_decision_variables(N, nx, nu, lbx, ubx):
    opti = cs.Opti()
    #X, U, S, W = [], [], [], [] #parte 1,2
    X, U = [], [] #parte 3
    for _ in range(N + 1):
        X += [opti.variable(nx)]
        opti.subject_to(opti.bounded(lbx, X[-1], ubx))
        # S += [opti.variable(1)] #tolgo per punto 3
        # opti.subject_to(opti.bounded(0, S[-1], 1))
    for _ in range(N):
        U += [opti.variable(nu)]
        # W += [opti.variable(1)]  #tolgo per punto 3

    return opti, X, U# , S, W #parte 3


def define_running_cost_and_dynamics(opti, X, U, N, dt, x_init,
                                     c_path, r_path, w_v, w_a, w_p,
                                     tau_min, tau_max, p_ref):
    # ho tolto S, W, w_w dalle variabili per parte 3
       
    # PATH CONSTRAINTS
    # TODO: Constrain the initial state X[0] to be equal to the initial condition x_init
    opti.subject_to(X[0]==x_init)

    # TODO: Initialize the path variable S[0] to 0.0
    #  opti.subject_to(S[0]==0.0) #parte 3

    # TODO: Constrain the final path variable S[-1] to be 1.0
    #  opti.subject_to(S[-1]==1.0) #parte 3
    #

    cost = 0.0
    for k in range(N):
        # TODO: Compute the end-effector position using forward kinematics
        q_k=X[k][:nq]
        ee_pos=fk(q_k)

        # TODO: Constrain ee_pos to lie on the desired path in x, y, z
        # p_x=c_path[0]+r_path*cs.cos(2*np.pi*S[k])
        # p_y=c_path[1]+r_path*0.5*cs.sin(4*np.pi*S[k])
        # p_z=c_path[2]
        # p_k=cs.vertcat(p_x,p_y,p_z)
       
        # opti.subject_to(ee_pos==p_k) #parte 1,2

        #costo di tracking
        p_ref_k = p_ref[:, k]
        cost += w_p * cs.sumsqr(ee_pos - p_ref_k)*dt #parte 3
        #opti.subject_to(ee_pos==p_ref_k) #prova constraint
        
        # TODO: Add velocity tracking cost term
        dq_k=X[k][nq:]
        cost+=w_v*cs.sumsqr(dq_k)*dt

        # TODO: Add actuation effort cost term
        u_k=U[k]
        cost+=w_a*cs.sumsqr(u_k)*dt

        # TODO: Add path progression speed cost term
        #w_k=W[k]
        # cost+=w_w*cs.sumsqr(w_k) #parte 1,2

        # TODO: Add discrete-time dynamics constraint
        # X[k+1] = X[k] + dt * f(X[k], U[k])
        # f(stato, controllo) -> f( (q, dq), u ) -> (dq, u)
        opti.subject_to(X[k+1]==X[k]+dt*f(X[k],U[k]))

        # TODO: Add path variable dynamics constraint
        # s_k+1 = s_k + dt * w_k
        # opti.subject_to(S[k+1]==S[k]+dt*W[k]) #parte 1,2

        # TODO: Constrain the joint torques to remain within [tau_min, tau_max]
        tau_k=inv_dyn(X[k],U[k])
        opti.subject_to(opti.bounded(tau_min,tau_k,tau_max))

        # Aggiunto costo sul tempo
        cost += (10**-1) * (dt**2)
        
        
    return cost

def define_terminal_cost_and_constraints(opti, X, w_p, x_init, c_path, r_path, w_final, p_ref):
# ho tolto S dalle variabili per parte 3 (neanche opti servirebbe più)

    # TODO: Compute the end-effector position at the final state
    q_N=X[-1][:nq]
    ee_pos_N=fk(q_N)

    # TODO: Constrain ee_pos to lie on the desired path in x, y, z at the end
    # p_x=c_path[0]+r_path*cs.cos(2*np.pi*S[-1])
    # p_y=c_path[1]+r_path*0.5*cs.sin(4*np.pi*S[-1])
    # p_z=c_path[2]
    # p_N=cs.vertcat(p_x,p_y,p_z)
       
    # opti.subject_to(ee_pos_N==p_N) #parte 1,2

    cost = 0
    cost += w_final * cs.sumsqr(X[-1] - x_init) #parte 2

    # AGGIUNTA parte 3 (costo di tracking terminale)
    p_ref_N = p_ref[:, -1] # Prendi l'ultimo punto del riferimento
    cost += w_p * cs.sumsqr(ee_pos_N - p_ref_N)

    #opti.subject_to(ee_pos_N==p_ref_N) #prova constraint
    return cost



def create_and_solve_ocp(N, nx, nq, lbx, ubx,
                         # dt,
                         x_init,
                         c_path, r_path, w_v, w_a, w_w, w_final, w_p,
                         tau_min, tau_max, p_ref):
    #opti, X, U, S, W = create_decision_variables(N, nx, nq, lbx, ubx) #parte 1,2
    opti, X, U = create_decision_variables(N, nx, nq, lbx, ubx) #parte 3

    # aggiunta per ottimizzazione tempi
    dt = opti.variable(1)
    opti.subject_to(opti.bounded(0.002, dt, 0.1))
    opti.set_initial(dt, 0.02) # Aiuta il solver dandogli un punto di partenza
   

    running_cost = define_running_cost_and_dynamics(opti, X, U, N, dt, x_init,
                                                    c_path, r_path, w_v, w_a,w_p,
                                                    tau_min, tau_max, p_ref)
    # ho tolto S, W, w_w dalle variabili per parte 3

    terminal_cost = define_terminal_cost_and_constraints(opti, X, w_p, x_init,c_path, r_path, w_final, p_ref)
    # ho tolto S dalle variabili per parte 3

    opti.minimize(running_cost + terminal_cost)

    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": 1e-4}

    opti.solver("ipopt", opts)

    t0 = time.time()
    sol = opti.solve()

    # aggiunta per ottimizzazione tempi
    dt_sol = sol.value(dt)
    print(f"Optimal dt: {dt_sol:.4f}s")
    print(f"Total time T: {N * dt_sol:.2f}s")

    print(f"Solver time: {time.time() - t0:.2f}s")
    return sol, X, U, dt_sol#, S, W


def extract_solution(sol, X, U, p_ref): #per parte 3 ho rimosso S e W come var. e ho messo p_ref
    x_sol = np.array([sol.value(X[k]) for k in range(N + 1)]).T
    ddq_sol = np.array([sol.value(U[k]) for k in range(N)]).T
    #s_sol = np.array([sol.value(S[k]) for k in range(N + 1)]).T #tolgo s, w per parte 3
    q_sol = x_sol[:nq, :]
    dq_sol = x_sol[nq:, :]
    #w_sol = np.array([sol.value(W[k]) for k in range(N)]).T #tolgo s, w per parte 3
    tau = np.zeros((nq, N))
    for i in range(N):
        tau[:, i] = inv_dyn(x_sol[:, i], ddq_sol[:, i]).toarray().squeeze()
    ee = np.zeros((3, N + 1))
    for i in range(N + 1):
        ee[:, i] = fk(x_sol[:nq, i]).toarray().squeeze()

    ee_des = np.zeros((3, N + 1))
    
    #for i in range(N + 1):
    #    ee_des[:, i] = np.array([c_path[0] + r_path*np.cos(2*np.pi*s_sol[i]),
    #                             c_path[1] + r_path*0.5*np.sin(4*np.pi*s_sol[i]),
    #                             c_path[2]])
        
    ee_des=p_ref # parte 3

    return q_sol, dq_sol, ddq_sol, tau, ee, ee_des#, s_sol, w_sol


# ====================== Simulation and Visualization ======================
r = RobotWrapper(robot.model, robot.collision_model, robot.visual_model)
simu = RobotSimulator(conf_ur5, r)
simu.init(q0, dq0)
simu.display(q0)

REF_SPHERE_RADIUS = 0.02
EE_REF_SPHERE_COLOR = np.array([1, 0, 0, .5])


def display_motion(q_traj, ee_des_traj, dt_sol):
    for i in range(N + 1):
        t0 = time.time()
        simu.display(q_traj[:, i])
        addViewerSphere(r.viz, f'world/ee_ref_{i}', REF_SPHERE_RADIUS, EE_REF_SPHERE_COLOR)
        applyViewerConfiguration(r.viz, f'world/ee_ref_{i}', ee_des_traj[:, i].tolist() + [0, 0, 0, 1.])
        t1 = time.time()
        if(t1-t0 < dt_sol):
            sleep(dt_sol - (t1-t0))



# ====================== Main Execution ======================
if __name__ == "__main__":
    print("Plotting reference infinity curve...")
    plot_infinity(0, 1)

    #default case
    #log_w_v, log_w_a, log_w_w, log_w_final = -3, -3, -2, 0

    #time optimization
    log_w_v, log_w_a, log_w_w, log_w_final = -6, -6, -6, 10

    log_w_p = 2 #Log of trajectory tracking cost 

    # REF TRAJECTORY PER PARTE 3
    # Creiamo la traiettoria di riferimento (s lineare da 0 a 1)
    s_ref = np.linspace(0, 1, N + 1)
    p_ref = np.zeros((3, N + 1))
    for k in range(N + 1):
        p_ref[0, k] = c_path[0] + r_path * np.cos(2 * np.pi * s_ref[k])
        p_ref[1, k] = c_path[1] + r_path * 0.5 * np.sin(4 * np.pi * s_ref[k])
        p_ref[2, k] = c_path[2]


    sol, X, U, dt_sol = create_and_solve_ocp( # ho tolto S e W dalla restituzione
        N, nx, nq, lbx, ubx,
        # dt,
        x_init, c_path, r_path,
        10**log_w_v, 10**log_w_a, 10**log_w_w, 10**log_w_final, 10**log_w_p,
        tau_min, tau_max, p_ref
    )
    q_sol, dq_sol, u_sol, tau, ee, ee_des = extract_solution(sol, X, U, p_ref)
 # ho tolto s_sol, w_sol  dalla restituzione e S e W come vars per parte 3

    print("Displaying robot motion...")
    for i in range(3):
        display_motion(q_sol, ee_des, dt_sol)

    # Plot results

    # TOLGO I PLOT DI s E w NELLA PARTE 3 PERCHè NON MI INTERESSANO PIù
    # tt = np.linspace(0, (N + 1) * dt, N + 1)
    tt = np.linspace(0, (N + 1) * dt_sol, N + 1) #per ottimizzazione tempi
    plt.figure(figsize=(10, 4))
    plt.plot([tt[0], tt[-1]], [0, 1], ':', label='straight line', alpha=0.7)
    plt.plot(tt, s_ref, label='s')
    plt.xlabel('Time [s]')
    plt.title('Evolution of s')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(ee_des[0,:].T, ee_des[1,:].T, 'r x', label='EE des', alpha=0.7)
    plt.plot(ee[0,:].T, ee[1,:].T, 'k o', label='EE', alpha=0.7)
    plt.xlabel('End-effector pos x [m]')
    plt.ylabel('End-effector pos y [m]')
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.plot(tt, ee_des[i,:].T, ':', label=f'EE des {i}', alpha=0.7)
        plt.plot(tt, ee[i,:].T, label=f'EE {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('End-effector pos [m]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(dq_sol.shape[0]):
        plt.plot(tt, dq_sol[i,:].T, label=f'dq {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 4))
    for i in range(q_sol.shape[0]):
        plt.plot([tt[0], tt[-1]], [q_sol[i,0], q_sol[i,0]], ':', label='straight line', alpha=0.7)
        plt.plot(tt, q_sol[i,:].T, label=f'q {i}', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.ylabel('Joint [rad]')
    plt.legend()
    plt.grid(True)

    prova1=np.ones(N)*max(tau_max)
    prova2=np.ones(N)*min(tau_min)
    plt.figure(figsize=(10, 4))
    for i in range(tau.shape[0]):
        plt.plot(tt[:-1], tau[i,:].T, label=f'tau {i}', alpha=0.7)
    plt.plot(tt[:-1],prova1, ':', label='tau_max' )
    plt.plot(tt[:-1],prova2, ':', label='tau_min' )
    plt.xlabel('Time [s]')
    plt.ylabel('Joint torque [Nm]')
    plt.legend()
    plt.grid(True)

    # TOLGO I PLOT DI s E w NELLA PARTE 3 PERCHè NON MI INTERESSANO PIù
    prova3=np.ones(N)*(s_ref[1]-s_ref[0])
    plt.figure(figsize=(10, 4))
    plt.plot(tt[:-1], prova3, label=f'w', alpha=0.7)
    plt.xlabel('Time [s]')
    plt.title('Evolution of w')
    plt.legend()
    plt.grid(True)
    plt.show()
