import torch
import numpy as np
from neural_network import NeuralNetwork

def check():
    print("--- Diagnostica Rete Neurale ---")
    # 1. Carica
    try:
        # Assicurati che hidden_size corrisponda al tuo train_model.py (128)
        net = NeuralNetwork(input_size=4, hidden_size=128, output_size=1, ub=10000.0)
        net.load_state_dict(torch.load('model.pt')['model'])
        net.eval()
        print("[OK] Modello caricato.")
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare il modello: {e}")
        return

    # 2. Test su stato Zero (Giù)
    x_down = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    val_down = net(x_down).item()
    print(f"Valore a riposo (Giù) [0,0,0,0]: {val_down:.4f}")
    
    # 3. Test su stato Target (Su)
    x_up = torch.tensor([[np.pi, 0.0, 0.0, 0.0]], dtype=torch.float32)
    val_up = net(x_up).item()
    print(f"Valore al target (Su) [pi,0,0,0]: {val_up:.4f}")
    
    if np.isnan(val_down) or np.isinf(val_down):
        print(">>> ALLARME: La rete predice NaN/Inf! Devi ri-allenare.")
    elif val_down < val_up:
         print(">>> ATTENZIONE: La rete dice che stare GIÙ costa meno che stare SU. Training fallito o dati errati.")
    else:
        print(">>> DIAGNOSI: La rete sembra sana (Giù > Su). Il problema è nel Solver.")

if __name__ == "__main__":
    check()