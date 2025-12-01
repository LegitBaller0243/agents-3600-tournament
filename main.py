import os
import torch
from alphazero.alphazero_config import AlphaZeroConfig
from alphazero_train import alphazero

if __name__ == "__main__":
    cfg = AlphaZeroConfig()

    final_network = alphazero(cfg)
    # Save trained weights for later use
    save_path = os.path.join(os.path.dirname(__file__), "trained_network_hour.pth")
    torch.save(final_network.state_dict(), save_path)
    print(f"Saved trained network to {save_path}")
