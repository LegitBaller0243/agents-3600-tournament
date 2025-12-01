from typing import Callable

from alphazero.game import Game
from alphazero.network import Network
from engine.game.enums import Direction, MoveType


def play_head_to_head(
    config,
    candidate_net: Network,
    best_net: Network,
    mcts_runner: Callable,
    candidate_is_player0: bool,
    verbose: bool = False,
    game_idx: int | None = None,
):
    """
    Play one deterministic game between candidate and best networks.
    mcts_runner should match run_mcts signature (config, game, network, add_noise=False).
    """
    # Force deterministic play during arena: no root noise, temperature = 0 (argmax).
    orig_sampling_moves = config.num_sampling_moves
    config.num_sampling_moves = 0

    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        to_play = game.to_play()
        use_candidate = (to_play == 0 and candidate_is_player0) or (to_play == 1 and not candidate_is_player0)
        net = candidate_net if use_candidate else best_net
        action, root = mcts_runner(config, game, net, add_noise=False)
        if verbose:
            total_visits = sum(child.visit_count for child in root.children.values())
            type_prior = {MoveType.PLAIN: 0.0, MoveType.EGG: 0.0, MoveType.TURD: 0.0}
            type_visit = {MoveType.PLAIN: 0.0, MoveType.EGG: 0.0, MoveType.TURD: 0.0}
            action_summaries = []
            for a, child in root.children.items():
                direction, mv = game.environment.decode_action(a)
                visit_frac = (child.visit_count / total_visits) if total_visits > 0 else 0.0
                type_prior[mv] += child.prior
                type_visit[mv] += visit_frac
                action_summaries.append(
                    (visit_frac, child.prior, a, direction, mv)
                )
            action_summaries.sort(key=lambda x: x[0], reverse=True)
            top_k = action_summaries[:3]
            top_fmt = "; ".join(
                f"a{a}:{direction.name}/{mv.name} visit={visit_frac:.2f} prior={prior:.2f}"
                for visit_frac, prior, a, direction, mv in top_k
            )
            type_mass = (
                f"visit={{plain={type_visit[MoveType.PLAIN]:.2f}, "
                f"egg={type_visit[MoveType.EGG]:.2f}, "
                f"turd={type_visit[MoveType.TURD]:.2f}}} "
                f"prior={{plain={type_prior[MoveType.PLAIN]:.2f}, "
                f"egg={type_prior[MoveType.EGG]:.2f}, "
                f"turd={type_prior[MoveType.TURD]:.2f}}}"
            )
            print(
                f"[Arena-Policy] game={game_idx if game_idx is not None else '?'} ply={len(game.history)} "
                f"to_play={to_play} choice={action} | {top_fmt} | {type_mass}"
            )
        game.apply(action)

    # Restore sampling settings after deterministic arena play.
    config.num_sampling_moves = orig_sampling_moves
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
        verbose = i < config.arena_log_games
        game = play_head_to_head(
            config,
            candidate_network,
            best_network,
            mcts_runner,
            candidate_as_player0,
            verbose=verbose,
            game_idx=i,
        )
        outcome = game.terminal_value(0)  # +1 if player0 wins, -1 if player0 loses, 0 tie
        if not candidate_as_player0:
            outcome *= -1  # flip perspective when candidate is player1
        if outcome > 0:
            candidate_wins += 1

        # Logging every 4th game to reduce noise.
        # Verbose logging disabled for now.

    return candidate_wins / games
