import numpy as np
import torch
from collections.abc import Callable
from typing import List, Tuple

# CS3600 engine
from engine.game.board import Board
from engine.game.enums import Direction, MoveType

# Your AlphaZero code
from alphazero.alphazero_config import AlphaZeroConfig
from alphazero.network import Network
from alphazero.game import Game
from alphazero_train import run_mcts   # uses your MCTS code


class PlayerAgent:
    """
    CS3600 tournament agent using your trained AlphaZero model + MCTS.
    """

    def __init__(self, board: Board, time_left: Callable):
        # Use the same config class, but reduce simulations for tournament time limits.
        self.config = AlphaZeroConfig()

        # Good tournament-safe values
        self.config.num_simulations = 50
        self.config.num_sampling_moves = 8

        # Load trained neural network
        self.device = torch.device("cpu")
        self.network = Network().to(self.device)

        try:
            state = torch.load("alphazero_first_edition.path", map_location=self.device)

            # If state saved as {"state_dict": ...}
            if isinstance(state, dict) and "state_dict" in state:
                self.network.load_state_dict(state["state_dict"])
            # If state saved directly as a state_dict
            elif isinstance(state, dict):
                self.network.load_state_dict(state)
            # If user saved the whole model (not ideal but possible)
            else:
                self.network = state.to(self.device)

            self.network.eval()
            print("[agent] Loaded trained model.")
        except Exception as e:
            print("[agent] ERROR loading model:", e)
            print("[agent] Falling back to random play.")
            self.network = None

    # -------------------------------------------------------------------
    # PLAY FUNCTION (called each turn)
    # -------------------------------------------------------------------
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Core move-selection logic:
        - Convert engine board → AlphaZero Game via from_board()
        - Use MCTS to select action
        - Decode action into (Direction, MoveType)
        """

        # If no trained model, choose random
        if self.network is None:
            moves = board.get_valid_moves()
            return moves[np.random.randint(len(moves))]

        # ---- Build AlphaZero Game state from CS3600 Board ----
        game = Game.from_board(board, sensor_data)

        # ---- Run MCTS to choose the best action ----
        action_index, root = run_mcts(self.config, game, self.network)

        # ---- Convert action index back to (Direction, MoveType) ----
        move = self.decode_action(action_index)

        # Safety check: engine ensures validity, but fallback if needed
        valid_moves = board.get_valid_moves()
        if move not in valid_moves:
            # If somehow invalid, return a random legal move
            return valid_moves[np.random.randint(len(valid_moves))]

        return move

    # -------------------------------------------------------------------
    # ACTION DECODING (must match Environment.encode logic)
    # -------------------------------------------------------------------
    def decode_action(self, action_index: int):
        direction_val = action_index // 3
        movetype_val = action_index % 3

        direction = Direction(direction_val)
        movetype = MoveType(movetype_val)

        return (direction, movetype)
