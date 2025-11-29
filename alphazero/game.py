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
    # Returns 1 if to_play wins, -1 if to_play loses, 0 if tie
    return self.environment.get_terminal_value(to_play)


  def legal_actions(self):
      return self.environment.get_legal_actions()

  def clone(self):
    cloned = Game.__new__(Game)

    cloned.history = list(self.history)
    cloned.child_visits = list(self.child_visits)
    cloned.environment = self.environment.clone()

    return cloned



  def apply(self, action):
    self.environment.apply_action(action)
    self.history.append(action)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
      return self.environment.get_image_tensor()

  def make_scalar_features(self):
    return self.environment.get_scalar_features()


  def make_target(self, state_index: int):
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])

  def to_play(self):
    return self.environment.to_play()
