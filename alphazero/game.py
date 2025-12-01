import sys
import os

# Add the engine directory to the path to import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.game.board import Board
from engine.game.game_map import GameMap
from engine.game.enums import Direction, MoveType, Result
from belief_system.environment import Environment


class Game(object):

  def __init__(self, history=None):
    # Initialize game map and board
    self.environment = Environment()

    # Initialize chickens if this is a new game (history is None or empty)
    if not history:
      # For a new game, we'd typically initialize spawns, but for AlphaZero
      # we might want to handle this differently. For now, we'll leave it
      # to be initialized externally if needed.
      pass
    
    # Store history and child visits for AlphaZero
    self.history = history or []
    self.child_visits = []

    self.num_actions = 12  

  def terminal(self):
    # Game specific termination rules.
      return self.environment.is_terminal()

  def terminal_value(self, to_play):
    # Game specific value.
    # Wins are +1.0. Losses are scaled by how far behind you are in eggs.
    board = self.environment.board
    base = self.environment.get_terminal_value(to_play)

    if base == 1:
      return 1.0

    if to_play == 0:
      my_eggs = board.chicken_player.get_eggs_laid()
      opp_eggs = board.chicken_enemy.get_eggs_laid()
    else:
      my_eggs = board.chicken_enemy.get_eggs_laid()
      opp_eggs = board.chicken_player.get_eggs_laid()

    if base == -1:
      diff = max(0, opp_eggs - my_eggs)
      total = max(1, my_eggs + opp_eggs)
      scaled = diff / total  # in (0,1]
      # Ensure some penalty for losses, but scale with closeness.
      return -max(0.2, min(1.0, scaled))

    # Tie or non-terminal
    return 0.0


  def legal_actions(self):
      return self.environment.get_legal_actions()

  def clone(self):
    cloned = Game(list(self.history))
    cloned.environment = self.environment.clone()
    cloned.child_visits = list(self.child_visits)
    return cloned


  def apply(self, action):
    self.environment.apply_action(action)
    self.history.append(action)

  def store_search_statistics(self, root, temperature: float = 1.0):
    sum_visits = sum(child.visit_count for child in root.children.values())
    if sum_visits == 0:
      self.child_visits.append([0 for _ in range(self.num_actions)])
      return

    temp = max(1e-3, float(temperature))
    adjusted = {}
    for a, child in root.children.items():
      # Sharpen or soften target distribution based on visit counts.
      adjusted[a] = (child.visit_count / sum_visits) ** (1.0 / temp)
    norm = sum(adjusted.values()) if adjusted else 1.0
    self.child_visits.append([
        (adjusted[a] / norm) if a in adjusted else 0.0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
      return self.environment.get_image_tensor()

  def make_scalar_features(self):
    return self.environment.get_scalar_features()


  def make_target(self, state_index: int):
    base_value = self.terminal_value(state_index % 2)
    shaping = 0.0
    try:
      # Small shaping toward egg advantage
      board = self.environment.board
      if state_index % 2 == 0:
        my_eggs = board.chicken_player.get_eggs_laid()
        opp_eggs = board.chicken_enemy.get_eggs_laid()
      else:
        my_eggs = board.chicken_enemy.get_eggs_laid()
        opp_eggs = board.chicken_player.get_eggs_laid()
      diff = my_eggs - opp_eggs
      shaping_coeff = getattr(self.environment, "shaping_reward_per_egg", None)
      if shaping_coeff is None:
        # Fallback to a default; environment may not carry config reference.
        shaping_coeff = 0.0
      shaping = shaping_coeff * diff
    except Exception:
      shaping = 0.0
    return (base_value + shaping,
            self.child_visits[state_index])

  def to_play(self):
    return self.environment.to_play()
