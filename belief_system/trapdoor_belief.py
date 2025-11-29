# trapdoor_belief.py

import sys
import os

# Add the engine directory to the path to import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from engine.game.game_map import prob_hear, prob_feel


class TrapdoorBelief:
    """
    Bayesian estimator for hidden trapdoor positions.
    
    Maintains:
      • white_prob : 8x8 belief map for white trapdoor
      • black_prob : 8x8 belief map for black trapdoor

    Provides:
      • initialization of priors
      • likelihood models
      • belief update via Bayes rule
      • tensor output for CNN
    """
    
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.white_prob = None
        self.black_prob = None
        self.initialize_priors()

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize_priors(self):
        """Set uniform or rule-based initial probability distributions."""
        # Based on trapdoor_manager: trapdoors are in center region with higher weight in inner region
        # White trapdoor = even parity (x+y) % 2 == 0
        # Black trapdoor = odd parity (x+y) % 2 == 1
        dim = self.board_size
        
        # Initialize with zeros
        white_prob = np.zeros((dim, dim))
        black_prob = np.zeros((dim, dim))
        
        # Set weights based on trapdoor_manager logic
        # Outer region (2:dim-2, 2:dim-2): weight 1.0
        # Inner region (3:dim-3, 3:dim-3): weight 2.0
        for x in range(dim):
            for y in range(dim):
                # Check if in valid region (center area)
                if 2 <= x < dim - 2 and 2 <= y < dim - 2:
                    weight = 1.0
                    if 3 <= x < dim - 3 and 3 <= y < dim - 3:
                        weight = 2.0
                    
                    # Assign to white (even) or black (odd) based on parity
                    if (x + y) % 2 == 0:
                        white_prob[x, y] = weight
                    else:
                        black_prob[x, y] = weight
        
        # Normalize each distribution
        white_sum = np.sum(white_prob)
        black_sum = np.sum(black_prob)
        if white_sum > 0:
            white_prob = white_prob / white_sum
        if black_sum > 0:
            black_prob = black_prob / black_sum
        
        self.white_prob = white_prob
        self.black_prob = black_prob

    # -----------------------------
    # Core Update
    # -----------------------------
    def update(self, player_position, white_observation, black_observation):
        """
        Perform Bayesian belief update for both white and black trapdoors.
        white_observation  = (heard_white, felt_white)
        black_observation  = (heard_black, felt_black)
        """
        # Update white trapdoor belief
        self.white_prob = self._bayes_update_one(self.white_prob, player_position, white_observation)
        
        # Update black trapdoor belief
        self.black_prob = self._bayes_update_one(self.black_prob, player_position, black_observation)

    def _bayes_update_one(self, belief_map, player_position, observation):
        """
        Perform Bayesian update on a single trapdoor's probability map.
        Uses Bayes rule: P(trapdoor|observation) ∝ P(observation|trapdoor) * P(trapdoor)
        """
        dim = self.board_size
        likelihood_map = np.zeros((dim, dim))
        
        # Compute likelihood for each possible trapdoor position
        for x in range(dim):
            for y in range(dim):
                likelihood_map[x, y] = self._likelihood(player_position, (x, y), observation)
        
        # Bayesian update: posterior ∝ likelihood * prior
        posterior = likelihood_map * belief_map
        
        # Normalize to get valid probability distribution
        posterior_sum = np.sum(posterior)
        if posterior_sum > 0:
            posterior = posterior / posterior_sum
        else:
            # If all probabilities are zero, keep prior (shouldn't happen normally)
            posterior = belief_map
        
        return posterior

    def _likelihood(self, player_pos, trap_pos, observation):
        """
        Compute likelihood P(observation | trapdoor at trap_pos).
        observation = (heard, felt) where heard and felt are booleans
        """
        heard, felt = observation
        delta_x = abs(player_pos[0] - trap_pos[0])
        delta_y = abs(player_pos[1] - trap_pos[1])
        
        # Probability of hearing given trapdoor at trap_pos
        hear_prob = prob_hear(delta_x, delta_y)
        if heard:
            hear_likelihood = hear_prob
        else:
            hear_likelihood = 1.0 - hear_prob
        
        # Probability of feeling given trapdoor at trap_pos
        feel_prob = prob_feel(delta_x, delta_y)
        if felt:
            feel_likelihood = feel_prob
        else:
            feel_likelihood = 1.0 - feel_prob
        
        # Joint likelihood (assuming independence)
        return hear_likelihood * feel_likelihood

    # -----------------------------
    # NN Output
    # -----------------------------
    def get_belief_tensor(self):
        """Return stacked white + black probability maps as a 2×8×8 tensor."""
        # Stack white and black probability maps
        # Shape: (2, 8, 8) where first channel is white, second is black
        return np.stack([self.white_prob, self.black_prob], axis=0)
    
    def clone(self):
        """Return a deep copy of the belief state."""
        cloned = TrapdoorBelief.__new__(TrapdoorBelief)
        cloned.board_size = self.board_size

        cloned.white_prob = self.white_prob.copy()
        cloned.black_prob = self.black_prob.copy()
        return cloned
