import os
from alphazero.alphazero_config import AlphaZeroConfig
from alphazero.network import Network
from alphazero.shared_storage import SharedStorage
from alphazero.game import Game
from alphazero.node import Node
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
from alphazero.utils import softmax_sample
from alphazero.replay_buffer import ReplayBuffer
from engine.game.enums import MoveType, Direction, loc_after_direction
from alphazero.arena import evaluate_against_best


def save_checkpoint(config: AlphaZeroConfig, network: Network, training_loop: int):
    """Persist the network so training can resume after crashes."""
    if config.checkpoint_interval <= 0 or not config.checkpoint_dir:
        return
    if (training_loop % config.checkpoint_interval) != 0:
        return

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.checkpoint_dir, f"alphazero_loop_{training_loop}.pth"
    )
    torch.save(
        {
            "training_loop": training_loop,
            "state_dict": network.state_dict(),
            "config": config.__dict__,
        },
        checkpoint_path,
    )
    print(f"[Checkpoint] Saved to {checkpoint_path}")


def alphazero(config: AlphaZeroConfig):
    storage = SharedStorage()
    network = Network()
    storage.save_network(0, network)

    replay_buffer = ReplayBuffer(config)
    best_network = storage.latest_network()

    for training_iter in range(config.training_loops):
        print(f"\n[AlphaZero] Training iteration {training_iter+1}/{config.training_loops}")
        
        num_games = config.selfplay_games_per_loop
        print(f"  [Self-Play] Generating {num_games} games...")
        run_selfplay(config, storage, replay_buffer, num_games)
        
        print(f"  [Training] Training network on {len(replay_buffer.buffer)} games...")
        avg_loss = train_network(config, storage, replay_buffer)
        print(f"  [Training] Average loss: {avg_loss:.4f}")

        candidate_network = storage.latest_network()
        # Evaluate against previous best and log win rate (no rejection threshold).
        if (training_iter + 1) % config.evaluation_interval == 0:
            win_rate = evaluate_against_best(config, best_network, candidate_network, run_mcts)
            print(f"  [Eval] Candidate vs Best win rate: {win_rate:.2%}")
            best_network = candidate_network
        else:
            print(f"  [Eval] Skipped (will evaluate every {config.evaluation_interval} loops)")

        save_checkpoint(config, candidate_network, training_iter + 1)
        
        print(f"[AlphaZero] Completed training iteration {training_iter+1}/{config.training_loops}")

    # Return best/latest network
    return best_network



##################################
####### Part 1: Self-Play ########

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer, num_games: int):
  for game_idx in range(num_games):
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)
    if (game_idx + 1) % 10 == 0:
      print(f"  [Self-Play] Generated {game_idx + 1}/{num_games} games")


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network, add_noise: bool = True):
  root = Node(0)
  evaluate(root, game, network)
  if add_noise:
    add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]

  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.items())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  # Get image tensor and scalar features
  image_tensor = game.make_image(-1)
  scalar_features = game.make_scalar_features()
  
  # Convert to numpy arrays if needed
  if isinstance(image_tensor, torch.Tensor):
    image_tensor = image_tensor.cpu().numpy()
  if isinstance(scalar_features, torch.Tensor):
    scalar_features = scalar_features.cpu().numpy()
  
  # Convert to torch tensors
  image_tensor = torch.FloatTensor(image_tensor)
  scalar_features = torch.FloatTensor(scalar_features)
  
  # Add batch dimension if needed
  if image_tensor.dim() == 3:
    image_tensor = image_tensor.unsqueeze(0)
  if scalar_features.dim() == 1:
    scalar_features = scalar_features.unsqueeze(0)
  
  # Set network to eval mode and get predictions
  device = next(network.parameters()).device
  image_tensor = image_tensor.to(device)
  scalar_features = scalar_features.to(device)
  
  value, policy = network.inference(image_tensor, scalar_features)
  
  # Convert to numpy for easier manipulation
  value = value.item() if isinstance(value, torch.Tensor) else float(value)
  policy = policy.squeeze(0).cpu().numpy() if isinstance(policy, torch.Tensor) else policy

  # Expand the node.
  node.to_play = game.to_play()
  legal_actions = game.legal_actions()
  if len(legal_actions) == 0:
    # No legal actions, return neutral value
    return 0.0
  
  # Normalize policy over legal actions only
  # Policy is a probability distribution over all actions (12 actions)
  policy_dict = {}
  board = game.environment.board
  acting_chicken = board.chicken_player if board.is_as_turn else board.chicken_enemy
  for action in legal_actions:
    if action < len(policy):
      p = float(policy[action])
      direction, mv = game.environment.decode_action(action)
      if mv == MoveType.EGG:
        p *= 3.0
      elif mv == MoveType.TURD:
        p *= 1.5
      elif mv == MoveType.PLAIN:
        # If a plain move steps onto an egg-able square for the acting chicken, boost it.
        dest = loc_after_direction(acting_chicken.get_location(), Direction(direction))
        if board.is_valid_cell(dest):
          parity_ok = (dest[0] + dest[1]) % 2 == acting_chicken.even_chicken
          blocked = (
              dest in board.eggs_player or dest in board.turds_player
              or dest in board.eggs_enemy or dest in board.turds_enemy
          )
          if parity_ok and not blocked:
            p *= 1.5
      # Add small floor so rare moves stay in play.
      policy_dict[action] = max(p, 0.0) + 1e-3

  policy_sum = sum(policy_dict.values())
  if policy_sum > 0:
    for action, p in policy_dict.items():
      node.children[action] = Node(p / policy_sum)
  else:
    # Uniform distribution if all zeros
    uniform_prob = 1.0 / len(legal_actions)
    for action in legal_actions:
      node.children[action] = Node(uniform_prob)
  
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    # Propagate value from the perspective of the player who just moved.
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = list(node.children.keys())
  if len(actions) == 0:
    return
  noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  # Normalize noise to form a proper Dirichlet distribution
  noise = noise / np.sum(noise)
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac




######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = storage.latest_network()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  network.to(device)
  network.train()
  
  # Use SGD with momentum (equivalent to TensorFlow's MomentumOptimizer)
  optimizer = optim.SGD(network.parameters(), lr=1e-2, momentum=config.momentum, weight_decay=config.weight_decay)
  
  total_loss = 0.0
  for i in range(config.training_steps):
    # Update learning rate according to schedule
    lr = get_learning_rate(i, config.learning_rate_schedule)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    
    batch = replay_buffer.sample_batch()
    if len(batch) > 0:
      loss = update_weights(optimizer, network, batch, config.weight_decay, device)
      total_loss += loss
    else:
      break  # No data to train on
  
  # Save updated network
  next_step = (max(storage._networks.keys()) + 1) if storage._networks else 0
  storage.save_network(next_step, network)
  
  avg_loss = total_loss / config.training_steps if config.training_steps > 0 else 0.0
  return avg_loss


def get_learning_rate(step: int, schedule: dict):
  """Get learning rate from schedule based on current step."""
  # Sort schedule keys in descending order
  sorted_steps = sorted(schedule.keys(), reverse=True)
  for schedule_step in sorted_steps:
    if step >= schedule_step:
      return schedule[schedule_step]
  # Default to first value if before all schedule points
  return schedule[sorted_steps[-1]] if sorted_steps else 2e-1


def update_weights(optimizer: optim.Optimizer, network: Network, batch,
                   weight_decay: float, device: torch.device):
  network.train()
  optimizer.zero_grad()
  
  if len(batch) == 0:
    return 0.0
  
  value_loss_fn = nn.MSELoss()
  
  # Collect all samples for batch processing
  images = []
  scalar_features_list = []
  target_values = []
  target_policies = []
  
  for item in batch:
    if len(item) == 3:
      image, scalar_feat, (target_value, target_policy) = item
    else:
      # Backward compatibility
      image, (target_value, target_policy) = item
      scalar_feat = np.zeros(4)  # Default scalar features
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
      image = image.cpu().numpy()
    if isinstance(scalar_feat, torch.Tensor):
      scalar_feat = scalar_feat.cpu().numpy()
    
    images.append(image)
    scalar_features_list.append(scalar_feat)
    target_values.append(target_value)
    target_policies.append(target_policy)
  
  # Convert to tensors and stack
  images_tensor = torch.FloatTensor(np.array(images)).to(device)
  scalar_features_tensor = torch.FloatTensor(np.array(scalar_features_list)).to(device)
  target_values_tensor = torch.FloatTensor(np.array(target_values)).to(device).unsqueeze(1)
  target_policies_tensor = torch.FloatTensor(np.array(target_policies)).to(device)
  
  # Forward pass
  values, policy_logits = network(images_tensor, scalar_features_tensor)
  
  # Compute losses
  value_loss = value_loss_fn(values, target_values_tensor)
  
  # Policy loss: target_policy is a distribution over actions
  # Use KL divergence between log_softmax(policy_logits) and target_policy
  log_probs = nn.functional.log_softmax(policy_logits, dim=1)
  eps = 1e-8
  # Type-weighted policy loss: egg/turd errors are penalized more.
  with torch.no_grad():
    action_indices = torch.arange(policy_logits.size(1), device=device)
    move_types = action_indices % 3  # 0 plain, 1 egg, 2 turd
    weights = torch.ones_like(log_probs)
    weights[:, move_types == 1] *= 2.0  # eggs
    weights[:, move_types == 2] *= 1.5  # turds
  kl_per_action = (target_policies_tensor + eps) * (torch.log(target_policies_tensor + eps) - log_probs)
  policy_loss = (weights * kl_per_action).sum() / weights.sum()

  total_loss = value_loss + policy_loss
  
  # Backward pass
  total_loss.backward()
  optimizer.step()
  
  return total_loss.item()

######### End Training ###########
##################################
