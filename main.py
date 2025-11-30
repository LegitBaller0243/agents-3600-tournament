from alphazero.alphazero_config import AlphaZeroConfig
from alphazero_train import alphazero
import os
import torch

if __name__ == "__main__":
    config = AlphaZeroConfig()
    final_network = alphazero(config)
    # Save trained weights for later use
    save_path = os.path.join(os.path.dirname(__file__), "trained_network_hour.pth")
    torch.save(final_network.state_dict(), save_path)
    print(f"Saved trained network to {save_path}")
