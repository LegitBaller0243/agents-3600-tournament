import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from alphazero.alphazero_config import AlphaZeroConfig
from alphazero.game import Game
from alphazero.node import Node
from alphazero.utils import softmax_sample
from alphazero.network import Network
from belief_system.environment import Environment
from belief_system.trapdoor_belief import TrapdoorBelief
from engine.game import board


WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "..", "alphazero_first_edition.pth")


class PlayerAgent:
    """
    /you may add functions, however, __init__ and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board: board.Board, time_left: Callable):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep a persistent belief over trapdoors and history for temperature logic.
        self.belief = TrapdoorBelief(board.game_map.MAP_SIZE)
        self.history: List[int] = []

        # Config tuned for inference (smaller search budget than training).
        self.config = AlphaZeroConfig()
        self.config.num_simulations = 60
        self.config.num_sampling_moves = 0  # deterministic after root search

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
        sims = self._simulations_for_time(remaining_time)

        # Run a lightweight MCTS search.
        root = Node(0)
        self._evaluate(root, game)
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

    def _simulations_for_time(self, remaining_time: Optional[float]) -> int:
        if remaining_time is None:
            return self.config.num_simulations
        if remaining_time < 2.0:
            return max(10, self.config.num_simulations // 3)
        if remaining_time < 5.0:
            return max(20, self.config.num_simulations // 2)
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
            priors[action] = float(policy[action]) if action < len(policy) else 0.0
        total = sum(priors.values())
        for action in legal_actions:
            prob = priors[action] / total if total > 0 else 1.0 / len(legal_actions)
            node.children[action] = Node(prob)
        return value

    def _backpropagate(self, search_path: List[Node], value: float, to_play: int):
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else (1 - value)
            node.visit_count += 1
