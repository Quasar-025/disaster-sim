"""
Flax Neural Networks for JAX PPO.

Actor-Critic with a CNN backbone matching the existing DisasterCNN
architecture (two conv layers → dense → separate actor/critic heads).
"""

from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic for the 6-channel grid observation.

    Input : ``[batch, 6, fov, fov]``   (channels-first)
    Output: ``(Categorical distribution, value [batch])``
    """

    action_dim: int = 6
    features_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Flax Conv expects channels-last → transpose from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # ---- CNN feature extractor (matches DisasterCNN) ----
        x = nn.Conv(features=32, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten spatial dims

        x = nn.Dense(self.features_dim)(x)
        x = nn.relu(x)

        # ---- Actor head ----
        actor = nn.Dense(128)(x)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim)(actor)

        # ---- Critic head ----
        critic = nn.Dense(128)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)

        pi = distrax.Categorical(logits=logits)
        return pi, jnp.squeeze(value, axis=-1)
