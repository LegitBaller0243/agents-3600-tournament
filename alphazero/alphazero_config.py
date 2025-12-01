class AlphaZeroConfig:
    def __init__(self):

        # ========================
        # Self-Play
        # ========================
        self.selfplay_games_per_loop = 16     # Very small, allows 8–12 loops total
        self.max_moves = 80
        self.num_sampling_moves = 6

        # Critical: FAST MCTS
        self.num_simulations = 40             # Fast AND enough to learn

        # ========================
        # Dirichlet Noise
        # ========================
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # ========================
        # PUCT
        # ========================
        self.pb_c_base = 50
        self.pb_c_init = 1.0

        # ========================
        # Training
        # ========================
        self.training_loops = 30   # You run until time runs out
        self.training_steps = 400     # 400 SGD steps per loop
        self.batch_size = 64
        self.window_size = 1000       # Small but sufficient for a tiny experiment
        self.momentum = 0.9
        self.weight_decay = 1e-4

        # ========================
        # Learning Rate Schedule
        # ========================
        self.learning_rate_schedule = {
            0:   3e-3,
            5:   2e-3,
            10:  1e-3,
        }

        # ========================
        # Arena
        # ========================
        self.evaluation_games = 10    # Keeps loop fast
        self.evaluation_win_threshold = 0.55
        self.evaluation_interval = 5

        # ========================
        # Checkpointing
        # ========================
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_interval = 5
