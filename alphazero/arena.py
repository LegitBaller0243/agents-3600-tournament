from typing import Callable

from alphazero.game import Game
from alphazero.network import Network


def play_head_to_head(
    config,
    candidate_net: Network,
    best_net: Network,
    mcts_runner: Callable,
    candidate_is_player0: bool,
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
        action, _ = mcts_runner(config, game, net, add_noise=False)
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
        game = play_head_to_head(config, candidate_network, best_network, mcts_runner, candidate_as_player0)
        outcome = game.terminal_value(0)  # +1 if player0 wins, -1 if player0 loses, 0 tie
        if not candidate_as_player0:
            outcome *= -1  # flip perspective when candidate is player1
        if outcome > 0:
            candidate_wins += 1
    return candidate_wins / games
