import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import time

try:
    from final_project_.neural_network.neural_network import NeuralNetwork
except ImportError:
    print("ERRORE: file non trovati")
    exit()

DATASET_NAME = "/home/claudia/orc/final_project_/dataset/dataset_pendulum.npz"        # Per il singolo
#DATASET_NAME = "/home/claudia/orc/final_project_/dataset/dataset_doublependulum.npz"    # Per il doppio


# --- PARAMETRI ---
LEARNING_RATE = 1e-3
EPOCHS = 1000
BATCH_SIZE = 64
TRAIN_RATIO = 0.8

# --- 1. CARICAMENTO DATI ---
def get_pendulum_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, DATASET_NAME)
    
    print(f"Loading data from {DATASET_NAME}...")
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print("ERRORE: File non trovato.")
        exit()

    X = data['inputs']
    Y = data['targets']

    X_tensor = torch.FloatTensor(X)
    Y_tensor = torch.FloatTensor(Y)

    nx = X.shape[1]
    if nx == 2:
        print(f"--> Rilevato PENDOLO SEMPLICE (nx={nx})")
        robot_name = "pendulum"
    elif nx == 4:
        print(f"--> Rilevato DOPPIO PENDOLO (nx={nx})")
        robot_name = "double_pendulum"
    else:
        print(f"--> Rilevato sistema generico (nx={nx})")
        robot_name = "generic"

    # Normalizzazione
    mean_X = X_tensor.mean(dim=0)
    std_X = X_tensor.std(dim=0); std_X[std_X < 1e-6] = 1.0
    mean_Y = Y_tensor.mean()
    std_Y = Y_tensor.std(); std_Y[std_Y < 1e-6] = 1.0

    X_norm = (X_tensor - mean_X) / std_X
    Y_norm = (Y_tensor - mean_Y) / std_Y

    return X_norm, Y_norm, mean_X, std_X, mean_Y, std_Y, nx, robot_name

# --- 2. CREAZIONE RETE ---
def create_network_and_optimizer(input_dim, lr):
    net = NeuralNetwork(input_size=input_dim, hidden_size=64, output_size=1, ub=10.0)
    # create the optimizer for training the network
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # use the Mean-Squared-Error (MSE) loss: (1/N) sum_i=1^N ||phi(in_i)-out_i||^2
    loss_func = nn.MSELoss()
    return net, optimizer, loss_func

# --- 3. TRAINING LOOP ---
def train_network(net, optim, loss_func, in_train, out_train, in_test, out_test, epochs, b_size):
    # Set the network in evaluation mode
    net.eval()

    # List of values taken by the loss on the training/test set during training
    loss_train = loss_func(net(in_train), out_train).item()
    losses_training = [loss_train]
    loss_test = loss_func(net(in_test), out_test).item()
    losses_test = [loss_test]

    train_size = out_train.shape[0]
    iters_per_epoch = int(train_size / b_size) # Number of iterations per epoch

    loss_min = loss_test # Keep track of the minimum value taken by the loss on the test set
    weights_min = copy.deepcopy(net.state_dict()) # Save best weights

    print(f"Starting training for {epochs} epochs...")

    for i in range(epochs):
        net.train() # Set the network in training mode
        
        total_loss = 0
        for k in range(iters_per_epoch):
            # Compute the loss on a randomly sampled mini-batch
            ind = torch.randint(train_size, size=(b_size,)) # Indici casuali
            
            # Forward pass
            prediction = net(in_train[ind])
            loss = loss_func(prediction, out_train[ind])
            total_loss += loss.item()
            
            # Backpropagation to compute the loss's gradient
            optim.zero_grad()
            loss.backward()
            # update the network's weights
            optim.step()

        # Store average loss for this epoch
        losses_training.append(total_loss / iters_per_epoch)

        net.eval()  # Set the model in evaluation mode
        with torch.no_grad():   # Disable gradient computation
            loss_test = loss_func(net(in_test), out_test).item()
        losses_test.append(loss_test)

        # Keep track of the weights corresponding to the minimum loss
        if loss_test < loss_min:
            loss_min = loss_test
            weights_min = copy.deepcopy(net.state_dict())
            
        if (i+1) % 100 == 0:
            print(f"Epoch {i+1}: Train Loss {losses_training[-1]:.5f}, Val Loss {losses_test[-1]:.5f}")

    # Return histories and the BEST weights
    return losses_training, losses_test, weights_min

# --- 4. MAIN ---
def main():
    # A. Carica Dati
    X, Y, mean_X, std_X, mean_Y, std_Y, nx, robot_name= get_pendulum_data()
    
    # Split Train/Test
    n_samples = len(X)
    n_train = int(TRAIN_RATIO * n_samples)
    indices = torch.randperm(n_samples)
    
    X_train = X[indices[:n_train]] # prendo 80% per train
    Y_train = Y[indices[:n_train]]
    X_test = X[indices[n_train:]]   # prendo restante 20% per il test
    Y_test = Y[indices[n_train:]]
    
    # B. Crea Rete
    input_dim = X.shape[1]
    net, optimizer, loss_func = create_network_and_optimizer(input_dim, LEARNING_RATE)
    
    # C. Lancia il Training
    start_time = time.time()
    train_hist, val_hist, best_weights = train_network(
        net, optimizer, loss_func, 
        X_train, Y_train, X_test, Y_test, 
        EPOCHS, BATCH_SIZE
    )
    end_time = time.time()
    training_duration = end_time - start_time

    print("-" * 30)
    print(f"TRAINING COMPLETATO.")
    print(f"Tempo impiegato: {training_duration:.2f} secondi ({training_duration/60:.2f} minuti)")
    print("-" * 30)

    # Carico i pesi migliori nella rete prima di salvare/plottare
    net.load_state_dict(best_weights)
    

    # D. Salvataggio
    output_filename = f"learned_value_{robot_name}.pth"
    torch.save({
        'model_state_dict': best_weights,
        'mean_X': mean_X, 'std_X': std_X,
        'mean_Y': mean_Y, 'std_Y': std_Y,
        'input_size': input_dim,
        'hidden_size': 64,
        'ub': 10.0,
        'robot_name': robot_name
    }, output_filename)
    print(f"Modello salvato in {output_filename}")

    # --- FUNZIONE FILTRO ---
    def low_pass_filter(data, alpha=0.1):
        """ Filtra la curva per vedere meglio il trend (senza rumore)"""
        if len(data) == 0: return []
        filtered_data = len(data)*[data[0]]
        for i in range(1, len(data)):
            filtered_data[i] = (1-alpha)*filtered_data[i-1] + alpha*data[i]
        return filtered_data

  
    # E. Plot Risultati

    plt.figure(figsize=(15,10))
    
    # Grafico Loss
    plt.subplot(2,2,1)
    plt.plot(train_hist, label="Train Loss", alpha=0.8, color='blue')
    plt.plot(val_hist, label="Test Loss", alpha=0.8, color='orange')
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Loss History")
    
    # 2. Loss History Filtrata
    train_hist_smooth = low_pass_filter(train_hist)
    val_hist_smooth = low_pass_filter(val_hist)

    plt.subplot(2, 2, 2)
    plt.plot(train_hist_smooth, label="Train Loss (Filt)", color='blue', linewidth=2)
    plt.plot(val_hist_smooth, label="Test Loss (Filt)", color='orange', linewidth=2)
    plt.yscale("log")
    plt.title("Filtered Loss History")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid(True)

    # Grafico Predizione vs Realtà (SCATTER PLOT) ---
    plt.subplot(2, 2, 3)
    net.eval()
    with torch.no_grad():
        pred = net(X_test)
    truth_real = Y_test.numpy() * std_Y.numpy() + mean_Y.numpy()
    pred_real = pred.numpy() * std_Y.numpy() + mean_Y.numpy()
    
    plt.scatter(truth_real, pred_real, alpha=0.5, s=10, c='blue', label='Real')
    
    # Linea Ideale Rossa
    min_v = min(truth_real.min(), pred_real.min())
    max_v = max(truth_real.max(), pred_real.max())
    plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Ideal')
    
    plt.title("Accuracy: prediction vs reality")
    plt.xlabel("Real Cost (CasADi)"); plt.ylabel("Prediction Cost (NN)")
    plt.legend(); plt.grid(True)


    # Grafico Value Function Landscape (HEATMAP)-
    plt.subplot(2, 2, 4)
    if nx >= 2:
        n_grid = 50
        # Creiamo una griglia di posizioni (q1, q2 o q, dq)
        x_range = np.linspace(-3.14, 3.14, n_grid)
        y_range = np.linspace(-3.14, 3.14, n_grid) # O velocità per pendolo singolo
        XX, YY = np.meshgrid(x_range, y_range)
        
        # Prepariamo input per la rete
        grid_input = np.zeros((n_grid*n_grid, nx))
        
        if nx == 2: # Pendolo Singolo (Pos vs Vel)
            grid_input[:, 0] = XX.ravel()
            grid_input[:, 1] = YY.ravel()
            xlabel, ylabel = "Position q", "Velocity dq"
        elif nx == 4: # Doppio Pendolo (q1 vs q2, con vel=0)
            grid_input[:, 0] = XX.ravel() # q1
            grid_input[:, 1] = YY.ravel() # q2
            # q3, q4 (velocità) restano a 0
            xlabel, ylabel = "Joint 1 (q1)", "Joint 2 (q2)"
            
        # Normalizza input griglia
        grid_tensor = torch.FloatTensor(grid_input)
        grid_norm = (grid_tensor - mean_X) / std_X
        
        with torch.no_grad():
            V_pred_norm = net(grid_norm)
        
        # Denormalizza output
        V_pred = V_pred_norm.numpy().reshape(n_grid, n_grid) * std_Y.numpy() + mean_Y.numpy()
        
        # Plot Heatmap
        plt.contourf(XX, YY, V_pred, levels=20, cmap='viridis')
        plt.colorbar(label="Value Function Cost")
        plt.title(f"Value Function Landscape ({robot_name})")
        plt.xlabel(xlabel); plt.ylabel(ylabel)

    plt.tight_layout() # Aggiusta gli spazi automaticamente

    
    try: plt.show()
    except: pass

if __name__ == "__main__":
    main()