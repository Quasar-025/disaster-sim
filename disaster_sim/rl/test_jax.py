"""
Testing script for the JAX PPO model.
Loads the trained weights and runs the agent in an interactive PyGame window.

Uses the JAX observation builder and step function to ensure perfect
observation match with the training environment.
"""

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import pygame

from disaster_sim.digital_twin.city_config import get_config, AGENT_TYPES, Terrain
from disaster_sim.digital_twin.city_generator import CityGenerator
from disaster_sim.digital_twin.world_state import WorldState
from disaster_sim.digital_twin.disaster_dynamics import DisasterDynamics
from disaster_sim.engine.physics import Action, PhysicsEngine
from disaster_sim.rl.jax_env import (
    EnvParams,
    EnvState,
    build_observation,
    make_params_for_agent,
)
from disaster_sim.rl.jax_networks import ActorCritic


def _build_jax_state_from_world(
    world: WorldState, agent_id: str, params: EnvParams,
    prev_row: int | None = None, prev_col: int | None = None,
) -> EnvState:
    """Convert CPU WorldState into a JAX EnvState for observation building."""
    agent = world.agents[agent_id]

    # Build explored map from agent's local_explored
    if agent.local_explored is not None:
        explored = jnp.array(agent.local_explored, dtype=jnp.bool_)
    else:
        explored = jnp.array(world.explored, dtype=jnp.bool_)

    # Build victim arrays (padded)
    victims = list(world.victims.values())
    n_v = min(len(victims), params.max_victims)
    v_row = np.zeros(params.max_victims, dtype=np.int32)
    v_col = np.zeros(params.max_victims, dtype=np.int32)
    v_alive = np.zeros(params.max_victims, dtype=bool)
    v_rescued = np.zeros(params.max_victims, dtype=bool)
    v_detected = np.zeros(params.max_victims, dtype=bool)
    v_severity = np.full(params.max_victims, 2, dtype=np.int32)
    v_mask = np.zeros(params.max_victims, dtype=bool)

    severity_map = {"critical": 0, "serious": 1, "stable": 2}

    for i, v in enumerate(victims[:params.max_victims]):
        v_row[i] = v.row
        v_col[i] = v.col
        v_alive[i] = v.is_alive
        v_rescued[i] = v.rescued
        v_detected[i] = v.detected
        v_severity[i] = severity_map.get(v.severity.value if hasattr(v.severity, 'value') else str(v.severity), 2)
        v_mask[i] = True

    # Build hospital arrays
    h_row = np.zeros(params.max_hospitals, dtype=np.int32)
    h_col = np.zeros(params.max_hospitals, dtype=np.int32)
    n_h = min(len(world.city.hospitals), params.max_hospitals)
    for i, (hr, hc) in enumerate(world.city.hospitals[:params.max_hospitals]):
        h_row[i] = hr
        h_col[i] = hc

    # Build charging station arrays
    c_row = np.zeros(params.max_charging, dtype=np.int32)
    c_col = np.zeros(params.max_charging, dtype=np.int32)
    n_c = min(len(world.city.charging_stations), params.max_charging)
    for i, (cr, cc) in enumerate(world.city.charging_stations[:params.max_charging]):
        c_row[i] = cr
        c_col[i] = cc

    pr = prev_row if prev_row is not None else int(agent.row)
    pc = prev_col if prev_col is not None else int(agent.col)

    return EnvState(
        grid=jnp.array(world.grid, dtype=jnp.int32),
        agent_row=jnp.int32(agent.row),
        agent_col=jnp.int32(agent.col),
        agent_prev_row=jnp.int32(pr),
        agent_prev_col=jnp.int32(pc),
        agent_battery=jnp.float32(agent.battery),
        agent_alive=jnp.bool_(agent.is_alive),
        agent_carrying=jnp.int32(-1),
        agent_cells_explored=jnp.int32(agent.cells_explored),
        explored=explored,
        victim_row=jnp.array(v_row),
        victim_col=jnp.array(v_col),
        victim_alive=jnp.array(v_alive),
        victim_rescued=jnp.array(v_rescued),
        victim_detected=jnp.array(v_detected),
        victim_severity=jnp.array(v_severity),
        victim_mask=jnp.array(v_mask),
        hospital_row=jnp.array(h_row),
        hospital_col=jnp.array(h_col),
        n_hospitals=jnp.int32(n_h),
        charging_row=jnp.array(c_row),
        charging_col=jnp.array(c_col),
        n_charging=jnp.int32(n_c),
        timestep=jnp.int32(world.timestep),
        total_collisions=jnp.int32(world.total_collisions),
        total_rescued=jnp.int32(world.total_victims_rescued),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/drone_jax_ppo.pkl")
    parser.add_argument("--preset", type=str, default="medium")
    parser.add_argument("--agent-type", type=str, default="drone")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from policy instead of greedy mode")
    args = parser.parse_args()

    # Load weights
    print(f"Loading weights from {args.model_path}...")
    try:
        with open(args.model_path, "rb") as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find model at {args.model_path}")
        print("Please ensure you've trained the model first.")
        return

    # Build JAX env params for observation building
    env_params = make_params_for_agent(args.preset, args.agent_type)

    # Initialize CPU environment for rendering
    pygame.init()
    config = get_config(args.preset)
    city = CityGenerator(config).generate()
    world = WorldState(city)
    physics = PhysicsEngine(world)
    dynamics = DisasterDynamics(world)

    # Pick agent
    agent_id = next(
        (aid for aid, a in world.agents.items() if a.agent_type == args.agent_type),
        list(world.agents.keys())[0]
    )

    # Set up renderer
    from disaster_sim.engine.renderer import CityRenderer
    renderer = CityRenderer(city, world)
    screen = pygame.display.set_mode((renderer.WINDOW_WIDTH, renderer.WINDOW_HEIGHT))
    pygame.display.set_caption("DisasterEnv — JAX Agent Test")
    clock = pygame.time.Clock()

    # Initialize network
    network = ActorCritic(action_dim=6, features_dim=128)

    rng = jax.random.PRNGKey(42)

    @jax.jit
    def get_action_greedy(params, obs_batch):
        pi, value = network.apply(params, obs_batch)
        return pi.mode()[0], value[0], pi.probs[0]

    @jax.jit
    def get_action_sample(params, obs_batch, rng):
        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=rng)
        return action[0], value[0], pi.probs[0]

    # Build initial observation using JAX obs builder
    jax_state = _build_jax_state_from_world(world, agent_id, env_params)
    obs = build_observation(jax_state, env_params)

    print("\n--- JAX AGENT TEST ---")
    print(f"Agent ID: {agent_id}")
    print(f"Using {'stochastic' if args.stochastic else 'greedy'} policy")
    print("Close the PyGame window to exit.")

    running = True
    total_reward = 0.0
    steps = 0
    prev_row, prev_col = int(world.agents[agent_id].row), int(world.agents[agent_id].col)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not running:
            break

        # Get action from JAX model
        obs_batched = jnp.expand_dims(obs, axis=0)
        if args.stochastic:
            rng, act_key = jax.random.split(rng)
            action, value, probs = get_action_sample(params, obs_batched, act_key)
        else:
            action, value, probs = get_action_greedy(params, obs_batched)
        action = int(action)

        # Step CPU environment
        agent = world.agents[agent_id]
        old_row, old_col = int(agent.row), int(agent.col)
        prev_explored = agent.cells_explored
        valid_action = physics.step(agent_id, action)
        physics.sync_communications()
        world.timestep += 1
        if world.timestep % 10 == 0:
            dynamics.step()

        actually_moved = (agent.row != old_row) or (agent.col != old_col)
        went_back = (agent.row == prev_row) and (agent.col == prev_col) and actually_moved

        # Compute reward (matching JAX logic)
        cells_revealed = agent.cells_explored - prev_explored
        reward = cells_revealed * 0.02 - 0.002  # Simplified reward for display
        if cells_revealed == 0:
            reward -= 0.003  # revisit penalty
        if not valid_action:
            reward -= 0.2
        if went_back:
            reward -= 0.03

        total_reward += reward
        steps += 1
        if actually_moved:
            prev_row, prev_col = old_row, old_col

        # Rebuild JAX observation from updated CPU state
        jax_state = _build_jax_state_from_world(
            world, agent_id, env_params,
            prev_row=prev_row, prev_col=prev_col,
        )
        obs = build_observation(jax_state, env_params)

        # Update prev position for next step
        prev_row, prev_col = old_row, old_col

        p_str = ", ".join([f"{p:.2f}" for p in np.array(probs)])
        print(f"Step {steps:3d} | Pos: ({agent.row:5.1f}, {agent.col:5.1f}) | Action: {action} | Value: {value:6.2f} | Reward: {reward:5.2f} | Probs: [{p_str}]")
        if steps in (36, 37, 38):
            comp = np.array(obs[5])
            print(f"Compass center 5x5:\n{comp[8:13, 8:13]}")
            terr = np.array(obs[0])
            print(f"Terrain center 5x5:\n{terr[8:13, 8:13]}")

        # Render
        screen.fill((20, 20, 30))
        city_surface = renderer.render_to_surface()
        view_width = renderer.WINDOW_WIDTH - renderer.INFO_PANEL_WIDTH
        view_height = renderer.WINDOW_HEIGHT
        city_w = city_surface.get_width()
        city_h = city_surface.get_height()
        offset_x = max(0, (view_width - city_w) // 2)
        offset_y = max(0, (view_height - city_h) // 2)
        screen.blit(city_surface, (offset_x, offset_y))
        renderer._render_info_panel(screen)
        pygame.display.flip()
        clock.tick(args.fps)

        terminated = not agent.is_alive or world.victims_remaining == 0
        truncated = world.timestep >= 1000

        if terminated or truncated:
            coverage = np.mean(world.explored) if hasattr(world.explored, '__len__') else world.coverage
            print(f"\nEpisode finished! Total steps: {steps}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Coverage: {coverage:.1%}")
            
            victims = world.victims.values()
            detected = sum(1 for v in victims if v.detected)
            total = len(victims)
            print(f"Victims detected: {detected} / {total}")
            print(f"Victims remaining: {world.victims_remaining}\n")

            # Restart
            config.seed = config.seed + 1 if hasattr(config, 'seed') else 1
            city = CityGenerator(config).generate()
            world = WorldState(city)
            physics = PhysicsEngine(world)
            dynamics = DisasterDynamics(world)
            renderer = CityRenderer(city, world)

            agent_id = next(
                (aid for aid, a in world.agents.items() if a.agent_type == args.agent_type),
                list(world.agents.keys())[0]
            )

            prev_row = int(world.agents[agent_id].row)
            prev_col = int(world.agents[agent_id].col)
            jax_state = _build_jax_state_from_world(world, agent_id, env_params)
            obs = build_observation(jax_state, env_params)

            total_reward = 0.0
            steps = 0
            time.sleep(1.0)

    pygame.quit()


if __name__ == "__main__":
    main()
