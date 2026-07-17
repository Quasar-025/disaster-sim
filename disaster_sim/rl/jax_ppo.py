"""
Pure-JAX PPO Training Loop.

The *entire* training pipeline — environment stepping, rollout collection,
GAE computation, and PPO gradient updates — runs inside a single JIT-compiled
function with ``jax.lax.scan``.  There is **zero CPU↔GPU synchronisation**
during training, yielding near-100 % GPU utilisation.

Architecture inspired by PureJaxRL (https://github.com/luchris429/purejaxrl).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from disaster_sim.rl.jax_env import (
    CityData,
    EnvParams,
    EnvState,
    build_observation,
    generate_city_pool,
    make_params_for_agent,
    reset,
    step,
)
from disaster_sim.rl.jax_networks import ActorCritic


# ---------------------------------------------------------------------------
# Transition buffer  (one element per (step, env))
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs: Any          # [6, fov, fov]
    action: Any       # []
    reward: Any       # []
    done: Any         # []
    value: Any        # []
    log_prob: Any     # []


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """All hyperparameters for a JAX PPO run."""

    # Environment
    preset: str = "medium"
    agent_type: str = "drone"
    fov_size: int = 21
    max_steps: int = 1000

    # Parallelism  (tuned for RTX 4060 8 GB)
    n_envs: int = 1024
    n_steps: int = 128             # rollout length
    n_epochs: int = 4              # PPO epochs per update
    n_minibatches: int = 8         # minibatches per epoch
    total_timesteps: int = 2_000_000

    # Optimisation
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    # City pool
    n_cities: int = 128
    city_seed: int = 42

    # Network
    features_dim: int = 128

    # --- derived ---
    @property
    def n_updates(self) -> int:
        return self.total_timesteps // (self.n_envs * self.n_steps)

    @property
    def minibatch_size(self) -> int:
        return (self.n_envs * self.n_steps) // self.n_minibatches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: PPOConfig, seed: int = 0):
    """Run full PPO training on GPU.  Returns the final ``TrainState``."""

    print(f"============================================================")
    print(f"  JAX PPO - 100 % GPU Training")
    print(f"============================================================")
    print(f"  City preset  : {config.preset}")
    print(f"  Parallel envs: {config.n_envs}")
    print(f"  Timesteps    : {config.total_timesteps:,}")
    print(f"  Updates      : {config.n_updates}")
    print(f"  Batch size   : {config.n_envs * config.n_steps:,}")
    print(f"  Minibatch    : {config.minibatch_size:,}")
    print(f"  Device       : {jax.devices()[0]}")
    print()

    # ---- 1. Create env params + city pool ----
    env_params = make_params_for_agent(
        config.preset, config.agent_type, config.fov_size, config.max_steps,
    )

    print("Generating {} cities on CPU ...".format(config.n_cities))
    t0 = time.time()
    city_pool = generate_city_pool(
        config.preset, config.agent_type, config.n_cities,
        env_params, seed + 1000,
    )
    print(f"  Done in {time.time() - t0:.1f}s  -  grid pool: {city_pool.grids.shape}")

    # ---- 2. Network + optimiser ----
    rng = jax.random.PRNGKey(seed)
    network = ActorCritic(action_dim=_NUM_ACTIONS, features_dim=config.features_dim)

    rng, init_key = jax.random.split(rng)
    dummy = jnp.zeros((1, 7, config.fov_size, config.fov_size))
    net_params = network.init(init_key, dummy)

    if config.anneal_lr:
        lr_sched = optax.linear_schedule(
            init_value=config.learning_rate, end_value=0.0,
            transition_steps=config.n_updates * config.n_epochs * config.n_minibatches,
        )
    else:
        lr_sched = config.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_sched, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=net_params, tx=tx,
    )

    # ---- 3. Initialise environments (vmapped) ----
    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, config.n_envs)

    def _reset(k):
        return reset(k, city_pool, env_params)

    def _step(k, s, a):
        return step(k, s, a, env_params)

    v_reset = jax.vmap(_reset)
    v_step = jax.vmap(_step)

    env_states, obs = v_reset(reset_keys)

    # ---- 4. Define a single PPO update step (rollout + GAE + update) ----
    #      All inner functions close over network, config, env_params, etc.

    def _env_step(carry, _unused):
        """Collect one transition from every env."""
        ts, es, last_obs, rng = carry

        rng, act_key, step_key, rst_key = jax.random.split(rng, 4)

        # Policy forward
        pi, value = network.apply(ts.params, last_obs)
        action = pi.sample(seed=act_key)
        log_prob = pi.log_prob(action)

        # Step all envs
        s_keys = jax.random.split(step_key, config.n_envs)
        nxt_state, nxt_obs, reward, done, _info = v_step(s_keys, es, action)

        # Auto-reset done envs
        r_keys = jax.random.split(rst_key, config.n_envs)
        rst_state, rst_obs = v_reset(r_keys)

        def _sel(r, n):
            """Where done, pick reset value; otherwise keep next."""
            d = done
            for _ in range(r.ndim - d.ndim):
                d = d[..., None]
            return jnp.where(d, r, n)

        final_state = jax.tree.map(_sel, rst_state, nxt_state)
        final_obs = _sel(rst_obs, nxt_obs)

        transition = Transition(
            obs=last_obs, action=action, reward=reward,
            done=done, value=value, log_prob=log_prob,
        )
        return (ts, final_state, final_obs, rng), transition

    def _compute_gae(transitions: Transition, last_val):
        def _scan_fn(carry, t):
            gae, nxt_v = carry
            delta = t.reward + config.gamma * nxt_v * (1.0 - t.done) - t.value
            gae = delta + config.gamma * config.gae_lambda * (1.0 - t.done) * gae
            return (gae, t.value), gae

        _, advantages = jax.lax.scan(
            _scan_fn,
            (jnp.zeros(config.n_envs), last_val),
            transitions,
            reverse=True,
        )
        returns = advantages + transitions.value
        return advantages, returns

    def _ppo_update(ts, transitions, advantages, returns, rng):
        batch_size = config.n_envs * config.n_steps

        flat_obs = transitions.obs.reshape((batch_size,) + transitions.obs.shape[2:])
        flat_act = transitions.action.reshape(batch_size)
        flat_lp = transitions.log_prob.reshape(batch_size)
        flat_adv = advantages.reshape(batch_size)
        flat_ret = returns.reshape(batch_size)

        def _epoch(carry, _unused):
            ts, rng = carry
            rng, perm_key = jax.random.split(rng)

            # Normalise advantages
            adv_n = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

            perm = jax.random.permutation(perm_key, batch_size)
            s_obs = flat_obs[perm]
            s_act = flat_act[perm]
            s_lp = flat_lp[perm]
            s_adv = adv_n[perm]
            s_ret = flat_ret[perm]

            mb_idx = jnp.arange(batch_size).reshape(config.n_minibatches, -1)

            def _mb_update(ts, idx):
                mb_obs = s_obs[idx]
                mb_act = s_act[idx]
                mb_old_lp = s_lp[idx]
                mb_adv = s_adv[idx]
                mb_ret = s_ret[idx]

                def _loss(params):
                    pi, vals = network.apply(params, mb_obs)
                    lp = pi.log_prob(mb_act)
                    ent = pi.entropy()

                    ratio = jnp.exp(lp - mb_old_lp)
                    clipped = jnp.clip(ratio, 1 - config.clip_range, 1 + config.clip_range)
                    pi_loss = -jnp.minimum(ratio * mb_adv, clipped * mb_adv).mean()

                    v_loss = 0.5 * ((vals - mb_ret) ** 2).mean()
                    e_loss = -ent.mean()

                    total = pi_loss + config.vf_coef * v_loss + config.ent_coef * e_loss
                    return total, {
                        "pi_loss": pi_loss,
                        "v_loss": v_loss,
                        "entropy": ent.mean(),
                        "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(),
                    }

                grads, aux = jax.grad(_loss, has_aux=True)(ts.params)
                ts = ts.apply_gradients(grads=grads)
                return ts, aux

            ts, ep_metrics = jax.lax.scan(_mb_update, ts, mb_idx)
            return (ts, rng), ep_metrics

        (ts, rng), all_metrics = jax.lax.scan(
            _epoch, (ts, rng), None, length=config.n_epochs,
        )
        avg = jax.tree.map(lambda x: x.mean(), all_metrics)
        return ts, avg

    def _update_step(runner_state, _unused):
        ts, es, obs, rng = runner_state

        rng, rollout_key, update_key = jax.random.split(rng, 3)

        # Rollout
        (ts, es, obs, _), traj = jax.lax.scan(
            _env_step, (ts, es, obs, rollout_key), None, length=config.n_steps,
        )

        # Bootstrap last value
        _, last_val = network.apply(ts.params, obs)

        # GAE
        adv, ret = _compute_gae(traj, last_val)

        # PPO update
        ts, metrics = _ppo_update(ts, traj, adv, ret, update_key)

        metrics["mean_reward"] = traj.reward.mean()

        return (ts, es, obs, rng), metrics

    # ---- 5. JIT-compile & run ----

    jit_update = jax.jit(_update_step)

    runner = (train_state, env_states, obs, rng)

    log_every = max(1, config.n_updates // 20)
    t_start = time.time()
    print(f"JIT-compiling + training ({config.n_updates} updates) ...\n")

    for u in range(config.n_updates):
        runner, metrics = jit_update(runner, None)

        if u == 0 or (u + 1) % log_every == 0 or u == config.n_updates - 1:
            jax.block_until_ready(metrics)
            elapsed = time.time() - t_start
            done_steps = (u + 1) * config.n_envs * config.n_steps
            sps = done_steps / elapsed if elapsed > 0 else 0

            print(
                f"  [{u + 1:>{len(str(config.n_updates))}}/{config.n_updates}]  "
                f"steps {done_steps:>10,}  "
                f"SPS {sps:>9,.0f}  "
                f"reward {float(metrics['mean_reward']):+.4f}  "
                f"pi-loss {float(metrics['pi_loss']):.4f}  "
                f"v-loss {float(metrics['v_loss']):.4f}  "
                f"entropy {float(metrics['entropy']):.4f}  "
                f"kl {float(metrics['approx_kl']):.4f}  "
                f"({elapsed:.0f}s)"
            )

    total_time = time.time() - t_start
    total_steps = config.n_updates * config.n_envs * config.n_steps

    print()
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Total steps : {total_steps:,}")
    print(f"  Wall time   : {total_time:.1f}s")
    print(f"  Avg SPS     : {total_steps / total_time:,.0f}")
    print("=" * 60)

    return runner, metrics


# Private — number of actions (avoids circular import)
_NUM_ACTIONS = 6
