# CS3600 Agent Strategy Options

This README summarizes three potential approaches for building our CS3600 tournament agent. All descriptions are intentionally concise and engineering-focused. Placeholder video/resource links are included where relevant.

---

## 1. MuZero-Style Agent (Neural + MCTS)

MuZero combines three neural networks (representation, dynamics, prediction) with Monte Carlo Tree Search to learn strategy directly from self-play. The system learns its own evaluation function and long-term planning behavior.

**Pros**
- Learns strategy automatically  
- No manual heuristics required  
- Excellent long-horizon planning  
- Strong portfolio / RL experience

**Cons**
- Most complex to implement  
- Requires GPU training + debugging  
- Higher development risk  

**Best Use Case**  
When we want a deep learning project that demonstrates advanced RL methods.

**Suggested Videos/Resources**  
- [MuZero Explanation](https://medium.com/@_michelangelo_/muzero-for-dummies-28fa076e781e)  
- [AlphaZero Concepts](https://www.youtube.com/watch?v=gsbkPpoxGQk&t=7s)

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
