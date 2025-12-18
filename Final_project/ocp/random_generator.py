import numpy as np
import matplotlib.pyplot as plt

def generate_random_initial_states(robot_model, n_samples=100, seed=None):
    """
    Genera N stati iniziali casuali rispettando i limiti del robot.
    Funziona sia per Pendolo Singolo (2 stati) che Doppio (4 stati).
    
    Args:
        robot_model: La classe del modello (deve avere .nx, .upperPositionLimit, etc.)
        n_samples: Quanti stati generare
        seed: Per rendere la generazione riproducibile (opzionale)
        
    Returns:
        X_init: Matrice numpy di dimensione (n_samples, nx)
    """
    if seed is not None:
        np.random.seed(seed)
        
    nx = robot_model.nx
    
    # 1. Costruisco i limiti globali dello stato x = [q, dq]
    # Assumo che il robot abbia limiti di posizione e velocità separati
    # Se il robot non ha limiti definiti, metto valori di default larghi
    
    try:
        # Limiti Posizione
        q_min = robot_model.lowerPositionLimit
        q_max = robot_model.upperPositionLimit
        
        # Limiti Velocità
        dq_max = robot_model.velocityLimit
        dq_min = -dq_max
        
        # lbx = [q_min..., dq_min...]
        lbx = np.concatenate([q_min, dq_min])
        ubx = np.concatenate([q_max, dq_max])
        
    except AttributeError:
        print("Attenzione: Il modello non ha limiti definiti. Uso limiti standard.")
        lbx = -5.0 * np.ones(nx)
        ubx =  5.0 * np.ones(nx)

    # 2. Generazione Casuale Uniforme
    # Crea una matrice (N, nx) dove ogni colonna scala tra min e max
    X_init = np.random.uniform(low=lbx, high=ubx, size=(n_samples, nx))
    
    #print(X_init)

    return X_init

