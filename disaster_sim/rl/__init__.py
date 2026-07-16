"""Reinforcement Learning environment and training utilities.

CPU-based (Gymnasium / Stable-Baselines3):
    env            — Gymnasium single-agent environment
    pz_env         — PettingZoo multi-agent environment
    observation    — Observation builder
    reward         — Reward calculator
    networks       — PyTorch CNN feature extractor
    train / train_ppo — SB3 PPO training scripts
    callbacks      — TensorBoard logging callbacks

GPU-accelerated (JAX — 100%% GPU utilisation):
    jax_env        — Vectorised JAX environment (state, reset, step)
    jax_networks   — Flax Actor-Critic network
    jax_ppo        — Pure-JAX PPO training loop
    train_jax      — CLI entry point for JAX training
"""
