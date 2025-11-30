from alphazero.alphazero_config import AlphaZeroConfig
from alphazero_train import alphazero
import torch

if __name__ == "__main__":
    config = AlphaZeroConfig()
    final_network = alphazero(config)
    torch.save(final_network.state_dict(), "alphazero_first_edition.pth")
