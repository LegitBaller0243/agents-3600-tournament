# CS3600 Agent Strategy Options

This README summarizes three potential approaches for building our CS3600 tournament agent. All descriptions are intentionally concise and engineering-focused. Placeholder video/resource links are included where relevant.

---
## 1. AlphaZero-Style Agent (Neural Network + MCTS)

AlphaZero uses a single neural network with two heads—a **policy head** and a **value head**—combined with Monte Carlo Tree Search. The game’s actual forward model (via `forecast_move()`) is used for all state transitions, so no learned dynamics model is required. The network improves through self-play, learning to evaluate states and choose strong actions without any handwritten heuristics.

**Pros**
- Much simpler than MuZero  
- No learned dynamics → faster, more stable training  
- Strong play through neural-guided MCTS  
- Fits realistically into a 1-week development timeline  

**Cons**
- Requires GPU time for self-play + training  
- Still more complex than pure MCTS  
- Network design and training loop must be implemented correctly  

**Best Use Case**  
When we want a powerful learning-based agent without the complexity and risk of full MuZero.

**Suggested Videos/Resources**  
- [AlphaZero Explained](https://www.youtube.com/watch?v=JxYX7NWWEAM)  
- [AlphaZero Paper Walkthrough](https://www.youtube.com/watch?v=ANDAk9yxZ1s)

---

## 2. Monte Carlo Tree Search (MCTS) + Handcrafted Evaluation

MCTS selectively explores promising actions using repeated simulations. Instead of a neural network, we provide a simple heuristic evaluation at leaf nodes.

**Pros**
- Very strong performance for this game  
- More efficient than minimax  
- Naturally handles uncertainty (trapdoor signals)  
- Moderate implementation effort  

**Cons**
- Requires a reasonably good heuristic  
- Slightly more complex than minimax  

**Best Use Case**  
If we want a high-performance agent with manageable engineering complexity.

**Suggested Videos/Resources**  
- [MCTS Basics](https://www.youtube.com/watch?v=lhFXKNyA0QA&t=30s)  


---

## 3. Minimax + Alpha-Beta Pruning

Traditional depth-limited search that evaluates all branches up to a fixed depth, pruning subtrees that cannot affect the optimal choice.

**Pros**
- Simple and deterministic  
- Easy to implement and debug  
- Strong baseline performance with a good evaluation function  

**Cons**
- Explores many unnecessary branches  
- Limited by search depth  
- Highly dependent on manual heuristics  

**Best Use Case**  
If we want a reliable, low-risk foundation or baseline agent.

---

Feel free to expand this README with implementation steps, architecture diagrams, or benchmarking once we select an approach.
