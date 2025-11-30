class AlphaZeroConfig:
    """
    AlphaZero configuration tuned for the Chicken Game (8×8, 40 moves/player).

    Notes:
    - Start with the SMALL configuration for debugging.
    - Move to MEDIUM once everything runs.
    - Use LARGE only on SLURM / multi-GPU training.
    """

    def __init__(self):
        # ========================
        # Self-Play Parameters
        # ========================
        # Targeting ~1 hour of CPU time: bump games/simulations for stronger search.
        self.selfplay_games_per_loop = 24  
        #   Small:  4–8
        #   Medium: 16–32
        #   Large:  64–256 (cluster)

        self.max_moves = 80                # 40 moves per player
        self.num_sampling_moves = 12       # Temperature softmax period
        self.num_simulations = 30         # MCTS sims per move

        # Ranges:
        #   Debug:    10–30
        #   Medium:   50–150
        #   Strong:   200–800 (expensive!)

        # ========================
        # Dirichlet Noise (exploration)
        # ========================
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # ========================
        # UCB Exploration Constants
        # ========================
        self.pb_c_base = 50             # Standard MuZero value
        self.pb_c_init = 1.25

        # ========================
        # Training
        # ========================
        self.training_loops = 40          
        # Recommended:
        #   Minimum to see improvement: ~30
        #   Medium strength: ~100–300
        #   Tournament-level: 500–2000 loops (cluster)

        self.training_steps = 30          
        # Steps per loop:
        #   CPU:      100–400
        #   GPU:      500–2000
        #   Cluster:  2000–10000

        self.batch_size = 32
        #   CPU:   64–128
        #   GPU:   128–512
        #   TPU:   512–2048

        self.window_size = 300           
        # Replay memory size (kept small to avoid stale data)

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # ========================
        # Learning Rate Schedule
        # ========================
        self.learning_rate_schedule = {
            0:   5e-3,
            10:  2e-3,
            25:  1e-3,
            35:  5e-4,
        }

        # ========================
        # Evaluation (anti-regression)
        # ========================
        self.evaluation_games = 12          # head-to-head games per loop
        self.evaluation_win_threshold = 0.55  # candidate must beat best with this win rate
