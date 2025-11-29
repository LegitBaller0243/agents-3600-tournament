import numpy as np
from .alphazero_config import AlphaZeroConfig

class ReplayBuffer(object):

  def __init__(self, config: AlphaZeroConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    if len(self.buffer) == 0:
      return []
    
    move_sum = float(sum(len(g.history) for g in self.buffer))
    if move_sum == 0:
      return []
    
    # Sample games weighted by number of moves
    probs = [len(g.history) / move_sum for g in self.buffer]
    game_indices = np.random.choice(
        len(self.buffer),
        size=min(self.batch_size, len(self.buffer)),
        p=probs,
        replace=True)
    games = [self.buffer[i] for i in game_indices]
    game_pos = [(g, np.random.randint(len(g.history))) for g in games if len(g.history) > 0]
    # Return (image, scalar_features, target_value, target_policy)
    return [(g.make_image(i), g.make_scalar_features(), g.make_target(i)) for (g, i) in game_pos]
