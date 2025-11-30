from alphazero.alphazero_config import AlphaZeroConfig
from alphazero_train import alphazero

if __name__ == "__main__":
    config = AlphaZeroConfig()
    final_network = alphazero(config)
