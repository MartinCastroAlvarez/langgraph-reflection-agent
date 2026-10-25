# Reflection in language agents

- Reflexion (Shinn et al., 2023) has an actor, an evaluator and a self-reflection model; verbal lessons from failed trials are kept in an episodic memory and added to the next trial's prompt.
- Reflexion improved pass@1 on the HumanEval coding benchmark by using unit test results as the evaluator signal.
- Language Agent Tree Search (LATS, Zhou et al., 2023) combines Monte Carlo tree search with LLM self-reflection: candidates are sampled, scored, and the most promising node is expanded.
- LATS backpropagates each new node's reward to its ancestors, and selects the next node with the upper confidence bound for trees (UCT).
- Self-critique without an external signal can reinforce a model's own mistakes; an external evaluator such as tests, a calculator or retrieval grounds the reflection.
