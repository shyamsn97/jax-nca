import functools
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from jax import lax


class SobelPerceptionNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x shape - BHWC

        num_channels = x.shape[-1]

        # 2D sobel kernels - IOHW layout

        x_sobel_kernel = jnp.zeros(
            (num_channels, num_channels, 3, 3), dtype=jnp.float32
        )
        x_sobel_kernel += (
            jnp.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])[
                jnp.newaxis, jnp.newaxis, :, :
            ]
            / 8.0
        )

        y_sobel_kernel = jnp.zeros(
            (num_channels, num_channels, 3, 3), dtype=jnp.float32
        )
        y_sobel_kernel += (
            jnp.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])[
                jnp.newaxis, jnp.newaxis, :, :
            ]
            / 8.0
        )
        x = jnp.transpose(x, [0, 3, 1, 2])  # N C H W

        x_out = lax.conv(
            x,  # lhs = NCHW image tensor
            x_sobel_kernel,  # rhs = OIHW conv kernel tensor
            (1, 1),  # window strides
            "SAME",
        )  # padding mode

        y_out = lax.conv(
            x,  # lhs = NCHW image tensor
            y_sobel_kernel,  # rhs = OIHW conv kernel tensor
            (1, 1),  # window strides
            "SAME",
        )  # padding mode

        out = jnp.concatenate([x, x_out, y_out], axis=1)
        return jnp.transpose(out, [0, 2, 3, 1])  # N H W C


class UpdateNet(nn.Module):

    num_channels: int

    @nn.compact
    def __call__(self, x):
        update_layer_1 = nn.Conv(
            features=64, kernel_size=(1, 1), strides=1, padding="VALID"
        )
        update_layer_2 = nn.Conv(
            features=64, kernel_size=(1, 1), strides=1, padding="VALID"
        )
        update_layer_3 = nn.Conv(
            features=self.num_channels,
            kernel_size=(1, 1),
            strides=1,
            padding="VALID",
            kernel_init=jax.nn.initializers.zeros,
            use_bias=False,
        )
        x = update_layer_1(x)
        x = nn.relu(x)
        x = update_layer_2(x)
        x = nn.relu(x)
        x = update_layer_3(x)
        return x


class TrainablePerception(nn.Module):
    num_channels: int

    @nn.compact
    def __call__(self, x):
        out = nn.Conv(
            features=self.num_channels * 3,
            kernel_size=(3, 3),
            use_bias=False,
            feature_group_count=self.num_channels,
        )(x)
        return out


@functools.partial(jax.jit, static_argnames=("apply_fn", "num_steps"))
def nca_multi_step(apply_fn, params, current_state: jnp.array, rng, num_steps: int):
    def forward(carry, inp):
        carry = apply_fn({"params": params}, carry, rng)
        return carry, carry

    x, outs = jax.lax.scan(forward, current_state, None, length=num_steps)
    return x, outs


class NCA(nn.Module):
    num_hidden_channels: int
    num_target_channels: int = 3
    alpha_living_threshold: float = 0.1
    cell_fire_rate: float = 1.0
    trainable_perception: bool = False
    alpha: float = 1.0

    """
        num_hidden_channels: Number of hidden channels for each cell to use
        num_target_channels: Number of target channels to be used
        alpha_living_threshold: threshold to determine whether a cell lives or dies
        cell_fire_rate: probability that a cell receives an update per step
        trainable_perception: if true, instead of using sobel filters use a trainable conv net
        alpha: scalar value to be multiplied to updates
    """

    @classmethod
    def create_seed(
        cls,
        num_hidden_channels: int,
        num_target_channels: int = 3,
        shape: Tuple[int] = (48, 48),
        batch_size: int = 1,
    ):
        seed = np.zeros((batch_size, *shape, num_hidden_channels + 3 + 1))
        w, h = seed.shape[1], seed.shape[2]
        seed[:, w // 2, h // 2, 3:] = 1.0
        return seed

    def setup(self):
        num_channels = 3 + self.num_hidden_channels + 1
        if self.trainable_perception:
            self.perception = TrainablePerception(num_channels)
        else:
            self.perception = SobelPerceptionNet()
        self.update_net = UpdateNet(num_channels)

    def alive(self, x, alpha_living_threshold: float):
        return (
            nn.max_pool(
                x[..., 3:4], window_shape=(3, 3), strides=(1, 1), padding="SAME"
            )
            > alpha_living_threshold
        )

    def get_stochastic_update_mask(self, x, rng, cell_fire_rate: float = 1.0):
        return jnp.array(np.random.uniform(size=x[..., :1].shape) <= cell_fire_rate)

    def __call__(self, x, rng):
        pre_life_mask = self.alive(x, self.alpha_living_threshold)

        perception_out = self.perception(x)
        update = self.alpha * jnp.reshape(self.update_net(perception_out), x.shape)

        if self.cell_fire_rate >= 1.0:
            stochastic_update_mask = self.get_stochastic_update_mask(
                x, rng, self.cell_fire_rate
            ).astype(float)
            x = x + update * stochastic_update_mask
        else:
            x = x + update

        post_life_mask = self.alive(x, self.alpha_living_threshold)

        life_mask = pre_life_mask & post_life_mask
        life_mask = life_mask.astype(float)

        return x * life_mask

    def save(self, params, path: str):
        bytes_output = serialization.to_bytes(params)
        with open(path, "wb") as f:
            f.write(bytes_output)

    def load(self, path: str):
        nca_seed = self.create_seed(
            self.num_hidden_channels, self.num_target_channels, batch_size=1
        )
        rng = jax.random.PRNGKey(0)
        init_params = self.init(rng, nca_seed, rng)["params"]
        with open(path, "rb") as f:
            bytes_output = f.read()
        return serialization.from_bytes(init_params, bytes_output)

    def multi_step(self, params, current_state: jnp.array, rng, num_steps: int = 2):
        return nca_multi_step(self.apply, params, current_state, rng, num_steps)

    def to_rgb(self, x: jnp.array):
        rgb, a = x[..., :3], jnp.clip(x[..., 3:4], 0.0, 1.0)
        rgb = jnp.clip(1.0 - a + rgb, 0.0, 1.0)
        return rgb
