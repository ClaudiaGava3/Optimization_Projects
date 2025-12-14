import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

def train():
    # 1. Caricamento Dati
    try:
        data = torch.load('dataset_double_pendulum.pt')
    except FileNotFoundError:
        print("Errore: Dataset non trovato. Esegui prima generate_data.py")
        return

    X = data['X']
    y = data['Y']
    print(f"Dataset caricato: {len(X)} campioni.")

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Modello (Più grande per il doppio pendolo)
    # IMPORTANTE: input_size=4, hidden_size=128
    model = NeuralNetwork(input_size=4, hidden_size=128, output_size=1, ub=10000.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 3. Training Loop
    epochs = 1500 # Un po' di più perché abbiamo più dati
    losses = []

    print(f"--- Inizio Training ({epochs} epoche) ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    # 4. Salvataggio
    torch.save({'model': model.state_dict()}, 'model.pt')
    print("Modello salvato in 'model.pt'")
    
    plt.plot(losses)
    plt.title("Training Loss")
    plt.yscale('log')
    plt.savefig('training_loss.png')

if __name__ == "__main__":
    train()