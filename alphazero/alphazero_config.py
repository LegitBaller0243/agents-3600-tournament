class AlphaZeroConfig:
    def __init__(self):

        # ========================
        # Self-Play
        # ========================
        self.selfplay_games_per_loop = 64     # More games per loop to stabilize policy/value heads
        self.max_moves = 80
        # Keep temperature-based sampling alive through the midgame.
        self.num_sampling_moves = 20
        self.midgame_temperature_start = 20
        self.midgame_temperature_span = 20

        # Critical: FAST MCTS
        self.num_simulations = 64             # More simulations for deeper search

        # ========================
        # Dirichlet Noise
        # ========================
        self.root_dirichlet_alpha = 0.15  # slightly lower to reduce injected noise
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
        # Target Sharpening
        # ========================
        # Temperature applied to MCTS visit counts before normalizing as training targets (<1.0 = sharper).
        self.training_target_temperature = 0.75
        # Encourage exploration during optimization; higher adds more entropy pressure.
        self.policy_entropy_coef = 2e-3

        # ========================
        # Arena
        # ========================
        self.evaluation_games = 40
        self.evaluation_win_threshold = 0.55
        self.evaluation_interval = 10
        # During the first N loops, auto-promote candidate to best after eval.
        self.early_promotion_loops = 10

        # ========================
        # Checkpointing
        # ========================
        self.checkpoint_dir = "checkpoints"
        self.checkpoint_interval = 5

        # ========================
        # Logging / Debug
        # ========================
        # Number of arena games to log detailed MCTS policy/visit info for.
        self.arena_log_games = 1
        # Keep in-memory network snapshots; set False for quick local tests.
        self.save_networks = True

        # ========================
        # Egg Prior/Reward Shaping
        # ========================
        # Global boost to egg priors (applied to network policy before normalization).
        self.egg_prior_boost = 1.5
        # Shaping reward per egg differential added to value targets.
        self.shaping_reward_per_egg = 0.1
        # Immediate value bonus for egg placements to speed up learning.
        self.egg_immediate_value = 0.3
        # Number of initial training loops to force egg moves when legal during self-play.
        self.scripted_egg_loops = 10
