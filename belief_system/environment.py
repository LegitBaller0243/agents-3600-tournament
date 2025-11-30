# environment.py

import sys
import os
from typing import Optional

import numpy as np

# Add the engine directory to the path to import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.game.board import Board
from engine.game.game_map import GameMap
from engine.game.enums import Direction, MoveType, Result
from engine.game.trapdoor_manager import TrapdoorManager

from .trapdoor_belief import TrapdoorBelief


class Environment:
    """
    Holds the full game state:
      • Staff Board object
      • Trapdoor belief state
      • Observation extraction (heard/felt)
      • Legal action generation
      • Image (CNN input) generation
      • Scalar feature generation
      • Action decoding & application
    """
    
    def __init__(self):
        # Follow the same initialization pattern as engine/gameplay.py
        self.game_map = GameMap()
        self.trapdoor_manager = TrapdoorManager(self.game_map)
        self.board = Board(self.game_map, time_to_play=360, build_history=False)
        self.belief = TrapdoorBelief()
        
        # Initialize spawns and trapdoors (same as TAs/teachers do)
        spawns = self.trapdoor_manager.choose_spawns()
        self.trapdoor_manager.choose_trapdoors()
        
        # Initialize chickens with spawns (same as TAs/teachers do)
        self.board.chicken_player.start(spawns[0], 0)  # Player A at spawns[0], even_chicken=0
        self.board.chicken_enemy.start(spawns[1], 1)    # Player B at spawns[1], even_chicken=1

    @classmethod
    def from_board(cls, board: Board, belief: Optional[TrapdoorBelief] = None):
        """
        Build an Environment around an existing Board copy.

        This is used by the tournament agent so we can run search on the
        referee-provided board while keeping our own trapdoor belief state.
        """
        env = cls.__new__(cls)  # bypass __init__ randomization
        env.game_map = board.game_map
        env.trapdoor_manager = None  # trapdoors are unknown in tournament play
        env.board = board.get_copy(build_history=False)
        env.belief = belief.clone() if belief is not None else TrapdoorBelief(board.game_map.MAP_SIZE)
        return env
    
    # -----------------------------
    # Core Game Control
    # -----------------------------
    def clone(self):
        """Return a deep copy of environment (board + belief)."""
        # Create a new environment but don't reinitialize (that would create new spawns/trapdoors)
        cloned = Environment.__new__(Environment)  # Create without calling __init__
        cloned.game_map = self.game_map  # GameMap is immutable, can share reference
        cloned.board = self.board.get_copy(build_history=False)
        cloned.belief = self.belief.clone()
        cloned.trapdoor_manager = self.trapdoor_manager  # Share reference to same trapdoor manager
        return cloned

    def is_terminal(self):
        """Return True if the game is over."""
        return self.board.is_game_over()

    def get_terminal_value(self, to_play):
        """Return +1/0/-1 game outcome from perspective of player to_play."""
        if not self.board.is_game_over():
            return 0
        
        winner = self.board.get_winner()
        if winner == Result.TIE:
            return 0
        
        # to_play: 0 = player A, 1 = player B
        # Result.PLAYER = 0 (player A), Result.ENEMY = 1 (player B)
        if (to_play == 0 and winner == Result.PLAYER) or (to_play == 1 and winner == Result.ENEMY):
            return 1
        else:
            return -1

    def to_play(self):
        """Return 0 or 1 depending on whose turn it is."""
        # is_as_turn: True = player A (0), False = player B (1)
        return 0 if self.board.is_as_turn else 1

    # -----------------------------
    # Move Handling
    # -----------------------------
    def apply_action(self, action: int, observations=None):
        """
        Decode integer action into (Direction, MoveType),
        apply move to Board, update trapdoor belief.
        """
        # Decode action
        direction, move_type = self.decode_action(action)
        
        # Get player position before move (for trapdoor observations)
        # Observations are from the square we're about to leave
        player_pos = self.board.chicken_player.get_location() if self.board.is_as_turn else self.board.chicken_enemy.get_location()
        
        # Get trapdoor observations from current position (before move)
        obs_to_use = observations
        if obs_to_use is None:
            obs_to_use = self.get_trapdoor_observations_at_position(player_pos)
        
        # If it's player B's turn, reverse perspective for apply_move
        needs_reverse = not self.board.is_as_turn
        if needs_reverse:
            self.board.reverse_perspective()
        
        # Apply move to board
        self.board.apply_move(direction, move_type, timer=0, check_ok=True)
        
        # Reverse back if needed
        if needs_reverse:
            self.board.reverse_perspective()
        
        # Update belief with observations from the position we just left
        if obs_to_use:
            white_obs, black_obs = obs_to_use
            self.belief.update(player_pos, white_obs, black_obs)

    def decode_action(self, action: int):
        """Convert action integer → (Direction, MoveType)."""
        # Action encoding: action = direction * 3 + move_type
        direction_val = action // 3
        move_type_val = action % 3
        return (Direction(direction_val), MoveType(move_type_val))

    def get_legal_actions(self):
        """Return a list of integers representing legal actions."""
        # Get valid moves from board based on whose turn it is
        enemy = not self.board.is_as_turn
        valid_moves = self.board.get_valid_moves(enemy=enemy)
        
        # Convert to action indices
        legal_actions = []
        for direction, move_type in valid_moves:
            action = direction.value * 3 + move_type.value
            legal_actions.append(action)
        return legal_actions

    # -----------------------------
    # Trapdoor Observation Handling
    # -----------------------------
    def get_trapdoor_observations(self):
        """
        Extract (heard, felt) signals for white & black trapdoors
        from the board API at the current player position.
        Returns ((heard_white, felt_white), (heard_black, felt_black)) or None if not available.
        """
        player_pos = self.board.chicken_player.get_location() if self.board.is_as_turn else self.board.chicken_enemy.get_location()
        return self.get_trapdoor_observations_at_position(player_pos)
    
    def get_trapdoor_observations_at_position(self, position):
        """
        Extract (heard, felt) signals for white & black trapdoors at a specific position.
        Returns ((heard_white, felt_white), (heard_black, felt_black)) or None if not available.
        """
        # If we have a trapdoor_manager with actual trapdoor positions, use it
        if self.trapdoor_manager is not None:
            observations = self.trapdoor_manager.sample_trapdoors(position)
            if len(observations) >= 2:
                return (observations[0], observations[1])
        
        # Otherwise, return None (observations not available)
        # In this case, belief won't be updated
        return None

    def update_belief(self):
        """Call belief.update() with decoded observations + player position."""
        player_pos = self.board.chicken_player.get_location() if self.board.is_as_turn else self.board.chicken_enemy.get_location()
        observations = self.get_trapdoor_observations()
        if observations:
            white_obs, black_obs = observations
            self.belief.update(player_pos, white_obs, black_obs)

    # -----------------------------
    # Neural Network Inputs
    # -----------------------------
    def get_image_tensor(self):
        """
        Return (5, 8, 8) tensor containing:
          - your board mask (eggs + turds + chicken)
          - opponent board mask (eggs + turds + chicken)
          - forbidden squares (enemy turd zones + enemy eggs)
          - white trapdoor belief map
          - black trapdoor belief map
        """
        dim = self.game_map.MAP_SIZE
        tensor = np.zeros((5, dim, dim))
        
        # Channel 0: Your board mask (eggs, turds, chicken)
        if self.board.is_as_turn:
            # Player A's turn - player is "player"
            for x, y in self.board.eggs_player:
                tensor[0, x, y] = 1.0
            for x, y in self.board.turds_player:
                tensor[0, x, y] = 1.0
            player_loc = self.board.chicken_player.get_location()
            tensor[0, player_loc[0], player_loc[1]] = 1.0
        else:
            # Player B's turn - player is "enemy", need to reverse perspective
            for x, y in self.board.eggs_enemy:
                tensor[0, x, y] = 1.0
            for x, y in self.board.turds_enemy:
                tensor[0, x, y] = 1.0
            enemy_loc = self.board.chicken_enemy.get_location()
            tensor[0, enemy_loc[0], enemy_loc[1]] = 1.0
        
        # Channel 1: Opponent board mask
        if self.board.is_as_turn:
            # Player A's turn - opponent is "enemy"
            for x, y in self.board.eggs_enemy:
                tensor[1, x, y] = 1.0
            for x, y in self.board.turds_enemy:
                tensor[1, x, y] = 1.0
            enemy_loc = self.board.chicken_enemy.get_location()
            tensor[1, enemy_loc[0], enemy_loc[1]] = 1.0
        else:
            # Player B's turn - opponent is "player"
            for x, y in self.board.eggs_player:
                tensor[1, x, y] = 1.0
            for x, y in self.board.turds_player:
                tensor[1, x, y] = 1.0
            player_loc = self.board.chicken_player.get_location()
            tensor[1, player_loc[0], player_loc[1]] = 1.0
        
        # Channel 2: Forbidden squares (enemy turd zones + enemy eggs)
        if self.board.is_as_turn:
            # Forbidden for player A = enemy turd zones + enemy eggs
            for x, y in self.board.eggs_enemy:
                tensor[2, x, y] = 1.0
            for x, y in self.board.turds_enemy:
                tensor[2, x, y] = 1.0
                # Add adjacent squares (turd zones)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < dim and 0 <= ny < dim:
                            tensor[2, nx, ny] = 1.0
        else:
            # Forbidden for player B = player turd zones + player eggs
            for x, y in self.board.eggs_player:
                tensor[2, x, y] = 1.0
            for x, y in self.board.turds_player:
                tensor[2, x, y] = 1.0
                # Add adjacent squares (turd zones)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < dim and 0 <= ny < dim:
                            tensor[2, nx, ny] = 1.0
        
        # Channels 3-4: Trapdoor belief maps
        belief_tensor = self.belief.get_belief_tensor()  # Shape: (2, 8, 8)
        tensor[3] = belief_tensor[0]  # White trapdoor belief
        tensor[4] = belief_tensor[1]  # Black trapdoor belief
        
        return tensor

    def get_scalar_features(self):
        """
        Return [your_turds_left,
                opponent_turds_left,
                your_turns_remaining,
                opponent_turns_remaining]
        """
        if self.board.is_as_turn:
            # Player A's turn
            your_turds = self.board.chicken_player.get_turds_left()
            opponent_turds = self.board.chicken_enemy.get_turds_left()
            your_turns = self.board.turns_left_player
            opponent_turns = self.board.turns_left_enemy
        else:
            # Player B's turn
            your_turds = self.board.chicken_enemy.get_turds_left()
            opponent_turds = self.board.chicken_player.get_turds_left()
            your_turns = self.board.turns_left_enemy
            opponent_turns = self.board.turns_left_player
        
        return [your_turds, opponent_turds, your_turns, opponent_turns]
