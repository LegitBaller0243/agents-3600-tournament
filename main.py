from alphazero.alphazero_config import AlphaZeroConfig
from alphazero import alphazero

if __name__ == "__main__":
    config = AlphaZeroConfig()
    final_network = alphazero(config)
