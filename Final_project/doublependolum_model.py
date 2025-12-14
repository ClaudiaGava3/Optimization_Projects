import numpy as np
import casadi as cs

class DoublePendulumModel:
    def __init__(self):
        # --- Parametri Fisici ---
        self.m1 = 1.0  # Massa link 1 [kg]
        self.m2 = 1.0  # Massa link 2 [kg]
        self.l1 = 1.0  # Lunghezza link 1 [m]
        self.l2 = 1.0  # Lunghezza link 2 [m]
        self.g = 9.81  # Gravità
        self.b = 0.1   # Attrito viscoso sui giunti

        # --- Dati Strutturali ---
        self.name = "double_pendulum"
        self.nq = 2    # Numero giunti (q1, q2)
        self.nx = 4    # Dimensione stato (q1, q2, dq1, dq2)
        self.nu = 2    # Dimensione input (tau1, tau2) o acc (ddq1, ddq2)
        
        # --- Limiti ---
        # Posizione: limitiamo a +/- 2 giri, ma potresti metterli infiniti
        self.lowerPositionLimit = np.array([-4*np.pi, -4*np.pi]) 
        self.upperPositionLimit = np.array([ 4*np.pi,  4*np.pi])
        
        # Velocità: [rad/s]
        self.velocityLimit = np.array([10.0, 10.0])       
        
        # Coppia Massima: [Nm]
        # Il doppio pendolo richiede più forza per essere alzato
        self.effortLimit = np.array([20.0, 20.0])          

    def get_dynamics_functions(self):
        """
        Restituisce le funzioni CasADi per la dinamica forward e inversa.
        Usa le equazioni di Lagrange: M*ddq + C*dq + G = tau
        """
        # Variabili simboliche (Vettori di dimensione 2)
        q   = cs.SX.sym('q', self.nq)
        dq  = cs.SX.sym('dq', self.nq)
        ddq = cs.SX.sym('ddq', self.nq) # Variabile di controllo (accelerazione)
        
        # Estrai le componenti per scrivere le equazioni leggibili
        q1, q2 = q[0], q[1]
        dq1, dq2 = dq[0], dq[1]
        ddq1, ddq2 = ddq[0], ddq[1]
        
        # Parametri brevi
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        
        # --- Matrice d'Inerzia M(q) ---
        # Elementi della matrice 2x2
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * cs.sin(q2) 
        # Nota: in molte convenzioni q=0 è giù, quindi cos(q2). 
        # Qui usiamo la convenzione standard robotica (spesso sin/cos variano).
        # Usiamo: q=0 verticale giù.
        M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * cs.cos(q2)
        M12 = m2 * l2**2 + m2 * l1 * l2 * cs.cos(q2)
        M21 = M12
        M22 = m2 * l2**2
        
        M = cs.vertcat(
            cs.horzcat(M11, M12),
            cs.horzcat(M21, M22)
        )

        # --- Matrice di Coriolis C(q, dq) ---
        h = m2 * l1 * l2 * cs.sin(q2)
        C11 = -h * dq2
        C12 = -h * (dq1 + dq2)
        C21 = h * dq1
        C22 = 0
        
        C = cs.vertcat(
            cs.horzcat(C11, C12),
            cs.horzcat(C21, C22)
        )
        
        # --- Vettore Gravità G(q) ---
        # Energia potenziale V = -mgl cos(theta) (se 0 è giù) -> G = dV/dq
        G1 = (m1 + m2) * g * l1 * cs.sin(q1) + m2 * g * l2 * cs.sin(q1 + q2)
        G2 = m2 * g * l2 * cs.sin(q1 + q2)
        G = cs.vertcat(G1, G2)
        
        # --- Attrito ---
        F = self.b * dq

        # --- DINAMICA INVERSA ---
        # tau = M*ddq + C*dq + G + F
        # Usiamo mtimes (@) per moltiplicazione matrice-vettore
        tau = M @ ddq + C @ dq + G + F
        
        # Funzione inv_dyn: [state, ddq] -> [tau]
        state = cs.vertcat(q, dq)
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

        # --- DINAMICA FORWARD ---
        # Serve per integrare lo stato: x_next = x + f(x, u) * dt
        # Poiché il controllo u nel tuo OCP è già ddq (accelerazione),
        # la funzione f è banale: dx = [dq, ddq]
        rhs = cs.vertcat(dq, ddq)
        f = cs.Function('f', [state, ddq], [rhs])

        return f, inv_dyn