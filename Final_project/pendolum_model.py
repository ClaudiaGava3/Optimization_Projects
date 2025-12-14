import numpy as np
import casadi as cs

class PendulumModel:
    def __init__(self):
        # --- Parametri Fisici ---
        self.m = 1.0   # Massa [kg]
        self.l = 1.0   # Lunghezza [m]
        self.g = 9.81  # Gravità [m/s^2]
        self.b = 0.1   # Attrito viscoso [N*m*s/rad]
        self.I = self.m * self.l**2 # Momento d'inerzia

        # --- Dati "Robot" (per mantenere compatibilità sintassi) ---
        self.name = "simple_pendulum"
        self.nq = 1  # Numero giunti
        self.nx = 2  # Dimensione stato (q, dq)
        self.nu = 1  # Dimensione input
        
        # Limiti (on vengono usati i constraints in realtà)
        self.velocityLimit = np.array([10.0])       # rad/s
        self.upperPositionLimit = np.array([2*np.pi]) 
        self.lowerPositionLimit = np.array([-2*np.pi])
        self.effortLimit = np.array([15.0])          # Nm (Max Torque)

    def get_dynamics_functions(self):
        """
        Restituisce le funzioni CasADi per la dinamica forward e inversa.
        Mantiene la firma che usavi nello script originale.
        """
        # Variabili simboliche
        q = cs.SX.sym('q', self.nq)
        dq = cs.SX.sym('dq', self.nq)
        ddq = cs.SX.sym('ddq', self.nq) # Qui ddq è l'input U del tuo OCP
        
        state = cs.vertcat(q, dq)
        
        # --- 1. Dinamica Inversa (Calcolo Tau date q, dq, ddq) ---
        # Equazione: I * ddq + b * dq + m * g * l * sin(q) = tau
        tau = self.I * ddq + self.b * dq + self.m * self.g * self.l * cs.sin(q)
        
        # Funzione inv_dyn: [state, ddq] -> [tau]
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

        # --- 2. Dinamica Forward (per l'integrazione dello stato) ---
        # Dato che nel tuo script U = ddq (accelerazione), la f(x,u) è semplice:
        # dq_next = dq + ddq * dt
        # q_next  = q  + dq * dt
        # Quindi rhs = [dq, ddq]
        rhs = cs.vertcat(dq, ddq)
        
        # Funzione f: [state, ddq] -> [rhs] (derivata dello stato)
        f = cs.Function('f', [state, ddq], [rhs])

        return f, inv_dyn