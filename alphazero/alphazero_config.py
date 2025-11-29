class AlphaZeroConfig(object):
  def __init__(self):
    self.selfplay_games_per_loop = 4      # You can increase later

    # Exploration
    self.num_sampling_moves = 10          # Early-game exploration
    self.max_moves = 80                   # Chicken game lasts 40 moves per player
    self.num_simulations = 50             # MCTS sims per move (reasonable for CPU)

    # Dirichlet noise
    self.root_dirichlet_alpha = 0.3
    self.root_exploration_fraction = 0.25

    # UCB exploration constants
    self.pb_c_base = 100                  # Much smaller
    self.pb_c_init = 1.25

    #### Training ####
    self.training_loops = 50              # 50 loops of (self-play + train)
    self.training_steps = 200             # SGD steps per loop
    self.batch_size = 64                  # Appropriate for CPU

    self.window_size = 20000              # Replay buffer size
    self.weight_decay = 1e-4
    self.momentum = 0.9

    # Simple learning rate schedule
    self.learning_rate_schedule = {
        0: 1e-2,
        20: 5e-3,
        40: 1e-3,
    }
