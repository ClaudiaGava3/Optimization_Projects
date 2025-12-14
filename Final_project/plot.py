import numpy as np
import matplotlib.pyplot as plt

# importo animazione
from matplotlib.animation import FuncAnimation

def animate_pendulum(time_array, q_array, length=1.0, dt=0.05):
    """
    Crea un'animazione del pendolo usando Matplotlib.
    """
    print("Generazione animazione...")
    fig, ax = plt.subplots(figsize=(6, 6))
        
    # Imposta i limiti degli assi (un po' più grandi della lunghezza del pendolo)
    limit = length * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Ottimizzazione Pendolo (N={len(time_array)})")

    # Elementi grafici: Asta (linea) e Massa (punto)
    line, = ax.plot([], [], 'o-', lw=4, color='blue', markersize=10) # 'o-' fa linea e punto finale
    trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)     # Scia rossa opzionale
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
    # Perno centrale fisso
    ax.plot(0, 0, 'ko', markersize=5)

    # Storia delle posizioni per la scia
    x_hist, y_hist = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text

    def update(frame):
        # Calcolo cinematica diretta: q=0 è GIÙ.
        # x = L * sin(q)
        # y = -L * cos(q)
        q = q_array[frame]
        x = length * np.sin(q)
        y = -length * np.cos(q)

        # Aggiorna asta
        line.set_data([0, x], [0, y])
            
        # Aggiorna scia
        x_hist.append(x)
        y_hist.append(y)
        trace.set_data(x_hist, y_hist)
            
        time_text.set_text(f'Time = {time_array[frame]:.2f}s')
        return line, trace, time_text

    # Crea l'animazione
    # interval è in millisecondi. dt * 1000 serve per farla andare a tempo reale.
    ani = FuncAnimation(fig, update, frames=len(time_array),
                        init_func=init, interval=dt*1000, blit=True)
        
    plt.show()


def animate_double_pendulum(time_array, q_array, l1=1.0, l2=1.0, dt=0.05):
    """
    Crea un'animazione del DOPPIO pendolo usando Matplotlib.
    
    Args:
        time_array: array dei tempi
        q_array: array numpy di forma (2, N) -> q[0,:]=q1, q[1,:]=q2
        l1: lunghezza primo link
        l2: lunghezza secondo link
        dt: timestep
    """
    print("Generazione animazione Doppio Pendolo...")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Imposta i limiti (somma delle lunghezze + margine)
    limit = (l1 + l2) * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Doppio Pendolo (N={len(time_array)})")

    # --- Elementi Grafici ---
    # 'o-' crea una linea con pallini ai vertici (Origine -> Gomito -> Punta)
    line, = ax.plot([], [], 'o-', lw=3, color='blue', markersize=8) 
    
    # Scia della punta finale (End-Effector)
    trace, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)
    
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    # Perno centrale fisso (Muro)
    ax.plot(0, 0, 'ks', markersize=8) # Quadrato nero

    # Storia delle posizioni per la scia
    x_hist, y_hist = [], []

    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text

    def update(frame):
        # 1. Estrai gli angoli al frame corrente
        # Assumiamo che q_array sia shape (2, N_samples)
        q1 = q_array[0, frame]
        q2 = q_array[1, frame]
        
        # 2. Cinematica Diretta (q=0 è GIÙ)
        # Link 1 (Gomito)
        x1 = l1 * np.sin(q1)
        y1 = -l1 * np.cos(q1)
        
        # Link 2 (Punta)
        # Nota: l'angolo assoluto del secondo link è (q1 + q2)
        x2 = x1 + l2 * np.sin(q1 + q2)
        y2 = y1 - l2 * np.cos(q1 + q2)

        # 3. Aggiorna Grafica
        # Passiamo le liste [x_start, x_mid, x_end] e [y_start, y_mid, y_end]
        line.set_data([0, x1, x2], [0, y1, y2])
        
        # Aggiorna scia (traccia solo la punta x2, y2)
        x_hist.append(x2)
        y_hist.append(y2)
        
        # Opzionale: tieni solo gli ultimi 100 punti per non appesantire
        if len(x_hist) > 200:
            x_hist.pop(0)
            y_hist.pop(0)
            
        trace.set_data(x_hist, y_hist)
        
        time_text.set_text(f'Time = {time_array[frame]:.2f}s')
        return line, trace, time_text

    # Crea l'animazione
    ani = FuncAnimation(fig, update, frames=len(time_array),
                        init_func=init, interval=dt*1000, blit=True)
    
    plt.show()


# --- 3. PLOT STATI INIZIALI ---
def plot_initial_states(X_init, robot_name="Robot"):
    n_samples, nx = X_init.shape
    plt.figure(figsize=(10, 6))
    
    if nx == 2:
        plt.scatter(X_init[:, 0], X_init[:, 1], alpha=0.6, c='blue', edgecolors='k')
        plt.xlabel("Posizione q [rad]")
        plt.ylabel("Velocità dq [rad/s]")
    elif nx >= 4:
        # Colora in base alla velocità media dei due giunti
        avg_vel = np.mean(np.abs(X_init[:, 2:]), axis=1)
        plt.scatter(X_init[:, 0], X_init[:, 1], c=avg_vel, cmap='viridis', alpha=0.7, edgecolors='k')
        plt.colorbar(label='Magnitudo Velocità Media')
        plt.xlabel("Posizione Giunto 1 [rad]")
        plt.ylabel("Posizione Giunto 2 [rad]")
        
    plt.title(f"Distribuzione Stati Iniziali - {robot_name} ({n_samples} samples)")
    plt.grid(True)
    plt.show()

# --- 4. FUNZIONE MAIN PLOTS (Logica corretta) ---
def PLOTS(DO_PLOTS, success_count, last_sol, N, dt, X, U, nq, nx, inv_dyn, q_des, tau_max, tau_min, robot, initial_conditions):
    
    # Esegui solo se richiesto e se c'è almeno una soluzione
    if DO_PLOTS and success_count > 0:
        print("Plotting results for the last successful sample...")
        
        time = np.linspace(0, N*dt, N+1)

        # Helper per estrarre dati puliti
        def get_data(var):
            val = last_sol.value(var)
            if hasattr(val, 'full'): val = val.full()
            elif hasattr(val, 'toarray'): val = val.toarray()
            return np.array(val).flatten()

        # Estrazione Dati
        x_sol = np.array([get_data(X[k]) for k in range(N+1)]).T
        u_sol = np.array([get_data(U[k]) for k in range(N)]).T
        
        # Gestione dimensioni se nq=1 o nq>1
        if x_sol.ndim == 1: x_sol = x_sol.reshape(nx, -1)
        if u_sol.ndim == 1: u_sol = u_sol.reshape(nq, -1)
        
        q_sol = x_sol[:nq,:]
        dq_sol = x_sol[nq:,:]
        
        # Calcolo Coppie
        tau_vals = np.zeros((nq, N))
        for k in range(N):
            val_tau = inv_dyn(x_sol[:,k], u_sol[:,k])
            if hasattr(val_tau, 'full'): val_tau = val_tau.full()
            tau_vals[:,k] = np.array(val_tau).flatten()

        # --- PLOT 1: POSIZIONE ---
        plt.figure(figsize=(10, 6))
        for j in range(nq):
            plt.plot(time, q_sol[j,:], label=f'q_{j+1} (pos)', linewidth=2)
            # Plot target (linea tratteggiata)
            plt.plot([0, time[-1]], [q_des[j], q_des[j]], '--', label=f'q_{j+1} target')
        plt.title("Joint Position")
        plt.xlabel('Time [s]'); plt.ylabel('Angle [rad]')
        plt.grid(True); plt.legend()

        # --- PLOT 2: VELOCITÀ ---
        plt.figure(figsize=(10, 6))
        for j in range(nq):
            plt.plot(time, dq_sol[j,:], label=f'dq_{j+1} (vel)', linewidth=2)
        plt.title("Joint Velocity")
        plt.xlabel('Time [s]'); plt.ylabel('Velocity [rad/s]')
        plt.grid(True); plt.legend()
        
        # --- PLOT 3: COPPIA ---
        plt.figure(figsize=(10, 6))
        for j in range(nq):
            plt.plot(time[:-1], tau_vals[j,:], label=f'tau_{j+1}', linewidth=2)
            # Limiti
            plt.plot([0, time[-1]], [tau_max[j], tau_max[j]], 'r--', alpha=0.3)
            plt.plot([0, time[-1]], [tau_min[j], tau_min[j]], 'r--', alpha=0.3)
        plt.title("Joint Torque")
        plt.xlabel('Time [s]'); plt.ylabel('Torque [Nm]')
        plt.grid(True); plt.legend()

        # --- SELEZIONE ANIMAZIONE ---
        # Logica: nq=1 -> Pendolo Singolo, nq=2 -> Doppio Pendolo
        if nq == 1:
            # Passiamo q_sol[0, :] perché è 1D
            animate_pendulum(time, q_sol[0,:], length=robot.l, dt=dt)
        elif nq == 2:
            # Passiamo tutto q_sol (2 righe)
            animate_double_pendulum(time, q_sol, l1=robot.l1, l2=robot.l2, dt=dt)
            
        # Plot distribuzione stati iniziali
        plot_initial_states(initial_conditions, robot.name)
        
        plt.show()