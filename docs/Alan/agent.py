import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

# Ensure project root is on sys.path so we can import our packages when run from docs/.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from alphazero.alphazero_config import AlphaZeroConfig
from alphazero.game import Game
from alphazero.node import Node
from alphazero.utils import softmax_sample
from alphazero.network import Network
from belief_system.environment import Environment
from belief_system.trapdoor_belief import TrapdoorBelief
from engine.game import board
from engine.game.enums import MoveType


# Weights file lives in the repository root.
WEIGHTS_FILE = os.path.join(ROOT_DIR, "alphazero_first_edition.pth")


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.device = torch.device("cpu")

        # Keep a persistent belief over trapdoors and history for temperature logic.
        self.belief = TrapdoorBelief(board.game_map.MAP_SIZE)
        self.history: List[int] = []

        # Config tuned for inference (smaller search budget than training) with root noise for exploration.
        self.config = AlphaZeroConfig()
        self.config.num_simulations = 30
        self.config.num_sampling_moves = 8  # early-game sampling

        self.network = self._load_network()
        self.network.to(self.device)
        self.network.eval()

    def play(
        self,
        board: board.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        # Update belief with real sensor data for this position.
        self._ingest_sensor_data(board, sensor_data)

        # Build an Environment/Game wrapper around the provided board.
        env = Environment.from_board(board, self.belief)
        game = Game(list(self.history))
        game.environment = env

        # Adapt search budget to remaining time.
        remaining_time = None
        try:
            remaining_time = time_left()
        except Exception:
            remaining_time = None
        moves_left = board.turns_left_player if board.is_as_turn else board.turns_left_enemy
        sims = self._simulations_for_time(remaining_time, moves_left)

        # Run a lightweight MCTS search.
        root = Node(0)
        self._evaluate(root, game)
        self._add_root_noise(root)
        for _ in range(sims):
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.expanded():
                action, node = self._select_child(node)
                scratch_game.apply(action)
                search_path.append(node)

            value = self._evaluate(node, scratch_game)
            self._backpropagate(search_path, value, scratch_game.to_play())

        action = self._select_action(game, root)
        self.history.append(action)

        # Translate integer action back to engine enums.
        direction, move_type = env.decode_action(action)

        # Safety: if chosen move is invalid, fall back to any legal move.
        if not board.is_valid_move(direction, move_type):
            valid = board.get_valid_moves()
            if valid:
                direction, move_type = valid[0]

        return (direction, move_type)

    # -----------------------------
    # Sensor + belief handling
    # -----------------------------
    def _ingest_sensor_data(self, board_obj: board.Board, sensor_data: List[Tuple[bool, bool]]):
        if not sensor_data or len(sensor_data) < 2:
            return
        location = board_obj.chicken_player.get_location() if board_obj.is_as_turn else board_obj.chicken_enemy.get_location()
        white_obs, black_obs = sensor_data[0], sensor_data[1]
        self.belief.update(location, white_obs, black_obs)

    # -----------------------------
    # Network + MCTS helpers
    # -----------------------------
    def _load_network(self) -> Network:
        network = Network()
        if os.path.exists(WEIGHTS_FILE):
            try:
                state = torch.load(WEIGHTS_FILE, map_location=self.device)
                network.load_state_dict(state)
            except Exception:
                # Fall back to randomly initialized network if load fails
                pass
        return network

    def _simulations_for_time(self, remaining_time: Optional[float], moves_left: Optional[int]) -> int:
        # time_left is a total clock (e.g., 360s for 40 moves). Scale sims by per-move budget.
        if remaining_time is None or moves_left is None:
            return self.config.num_simulations

        per_move = remaining_time / max(1, moves_left)
        if per_move < 1.0:
            return max(10, self.config.num_simulations // 3)
        if per_move < 2.0:
            return max(16, self.config.num_simulations // 2)
        if per_move < 4.0:
            return max(20, int(self.config.num_simulations * 0.75))
        if remaining_time < 60.0:
            return max(22, int(self.config.num_simulations * 0.9))
        return self.config.num_simulations

    def _select_action(self, game: Game, root: Node) -> int:
        visit_counts = [(child.visit_count, action) for action, child in root.children.items()]
        if not visit_counts:
            # Fallback to any legal move if search failed
            legal = game.legal_actions()
            return np.random.choice(legal)
        if len(game.history) < self.config.num_sampling_moves:
            _, action = softmax_sample(visit_counts)
        else:
            _, action = max(visit_counts)
        return action

    def _select_child(self, node: Node):
        def ucb_score(parent: Node, child: Node):
            pb_c = (
                np.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base)
                + self.config.pb_c_init
            )
            pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
            prior_score = pb_c * child.prior
            value_score = child.value()
            return prior_score + value_score

        _, action, child = max(
            (ucb_score(node, child), action, child) for action, child in node.children.items()
        )
        return action, child

    def _evaluate(self, node: Node, game: Game):
        image_tensor = game.make_image(-1)
        scalar_features = game.make_scalar_features()

        image_tensor = torch.as_tensor(image_tensor, dtype=torch.float32, device=self.device)
        scalar_features = torch.as_tensor(scalar_features, dtype=torch.float32, device=self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if scalar_features.dim() == 1:
            scalar_features = scalar_features.unsqueeze(0)

        value, policy = self.network.inference(image_tensor, scalar_features)
        value = value.item() if isinstance(value, torch.Tensor) else float(value)
        policy = policy.squeeze(0).detach().cpu().numpy()

        node.to_play = game.to_play()
        legal_actions = game.legal_actions()
        if not legal_actions:
            return 0.0

        priors = {}
        for action in legal_actions:
            prior = float(policy[action]) if action < len(policy) else 0.0
            # Heuristic boost for egg moves to promote scoring.
            _, mv_type = env.decode_action(action)
            if mv_type == MoveType.EGG:
                prior *= 1.3
            priors[action] = prior
        total = sum(priors.values())
        for action in legal_actions:
            # Keep a floor so every legal action (including eggs) remains explorable.
            prob = priors[action] / total if total > 0 else 0.0
            prob = prob + 1e-3
            node.children[action] = Node(prob)
        # Renormalize priors to sum to 1
        norm = sum(child.prior for child in node.children.values())
        if norm > 0:
            for child in node.children.values():
                child.prior /= norm
        return value

    def _backpropagate(self, search_path: List[Node], value: float, to_play: int):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1

    def _add_root_noise(self, root: Node):
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.gamma(self.config.root_dirichlet_alpha, 1, len(actions))
        noise = noise / np.sum(noise)
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            root.children[a].prior = root.children[a].prior * (1 - frac) + n * frac
