"""
Flax Neural Networks for JAX PPO.

Actor-Critic with a CNN backbone. Uses strided convolutions instead of
max-pooling to preserve fine-grained spatial signals (compass channel).
Layer normalization improves training stability.
"""

from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


class ActorCritic(nn.Module):
    """Shared-backbone Actor-Critic for the 7-channel grid observation.

    Input : ``[batch, 7, fov, fov]``   (channels-first)
    Output: ``(Categorical distribution, value [batch])``
    """

    action_dim: int = 6
    features_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Flax Conv expects channels-last → transpose from NCHW to NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))

        # ---- CNN feature extractor ----
        # Conv1: strided conv replaces conv+max_pool to preserve compass signal
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Conv2: regular conv for higher-level features
        x = nn.Conv(features=64, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # flatten spatial dims

        x = nn.Dense(self.features_dim)(x)
        x = nn.LayerNorm()(x)
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
