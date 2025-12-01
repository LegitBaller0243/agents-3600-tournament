from typing import Callable

from alphazero.game import Game
from alphazero.network import Network
from engine.game.enums import MoveType


def play_head_to_head(
    config,
    candidate_net: Network,
    best_net: Network,
    mcts_runner: Callable,
    candidate_is_player0: bool,
    verbose: bool = False,
):
    """
    Play one deterministic game between candidate and best networks.
    mcts_runner should match run_mcts signature (config, game, network, add_noise=False).
    """
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        to_play = game.to_play()
        use_candidate = (to_play == 0 and candidate_is_player0) or (to_play == 1 and not candidate_is_player0)
        net = candidate_net if use_candidate else best_net
        if verbose:
            legal_actions = game.legal_actions()
            legal_counts = {"plain": 0, "egg": 0, "turd": 0}
            for a in legal_actions:
                _, mv = game.environment.decode_action(a)
                if mv == MoveType.PLAIN:
                    legal_counts["plain"] += 1
                elif mv == MoveType.EGG:
                    legal_counts["egg"] += 1
                elif mv == MoveType.TURD:
                    legal_counts["turd"] += 1
        action, root = mcts_runner(config, game, net, add_noise=False)
        if verbose:
            prior_counts = {"plain": 0.0, "egg": 0.0, "turd": 0.0}
            for a, child in root.children.items():
                _, mv = game.environment.decode_action(a)
                if mv == MoveType.PLAIN:
                    prior_counts["plain"] += child.prior
                elif mv == MoveType.EGG:
                    prior_counts["egg"] += child.prior
                elif mv == MoveType.TURD:
                    prior_counts["turd"] += child.prior
            print(
                f"[Arena-Verbose] ToPlay={to_play} | Legal counts {legal_counts} | Prior mass {prior_counts}"
            )
        game.apply(action)
    return game


def evaluate_against_best(config, best_network: Network, candidate_network: Network, mcts_runner: Callable) -> float:
    """
    Run head-to-head matches to decide if the candidate improves on best_network.
    Returns win rate for the candidate.
    """
    candidate_wins = 0
    games = max(1, config.evaluation_games)
    for i in range(games):
        candidate_as_player0 = (i % 2 == 0)
        verbose = False  # logging disabled
        game = play_head_to_head(config, candidate_network, best_network, mcts_runner, candidate_as_player0, verbose=verbose)
        outcome = game.terminal_value(0)  # +1 if player0 wins, -1 if player0 loses, 0 tie
        if not candidate_as_player0:
            outcome *= -1  # flip perspective when candidate is player1
        if outcome > 0:
            candidate_wins += 1

        # Logging every 4th game to reduce noise.
        # Verbose logging disabled for now.

    return candidate_wins / games
