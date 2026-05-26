import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax.linen.initializers import orthogonal, zeros

from strelax.algorithms import AVG, AVGConfig
from strelax.environments import environment
from strelax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from strelax.networks import heads

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "brax::halfcheetah"

gamma = 0.99
alpha = 0.07
actor_lr = 0.0063
critic_lr = 0.0087
beta1 = 0.0
beta2 = 0.999
eps = 1e-8
n_hid = 256

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

action_dim = env.action_space(env_params).shape[0]


def l2_normalize(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def feature_extractor(x: jax.Array) -> jax.Array:
    x = nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(x)
    x = nn.leaky_relu(x)
    x = nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(x)
    x = nn.leaky_relu(x)
    return l2_normalize(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> object:
        features = feature_extractor(obs)
        return heads.SquashedGaussian(
            action_dim=self.action_dim, kernel_init=orthogonal()
        )(features)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = feature_extractor(jnp.concatenate([obs, action], axis=-1))
        return heads.ContinuousQNetwork(kernel_init=orthogonal())(features, action)


config = AVGConfig(
    num_envs=1,
    gamma=gamma,
    alpha=alpha,
    trace_lambda=0.0,
)

actor_network = Actor(action_dim=action_dim)
critic_network = Critic()

actor_optimizer = optax.adam(actor_lr, b1=beta1, b2=beta2, eps=eps)
critic_optimizer = optax.adam(critic_lr, b1=beta1, b2=beta2, eps=eps)

agent = AVG(
    config,
    env,
    env_params,
    actor_network,
    critic_network,
    actor_optimizer,
    critic_optimizer,
)


init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

train_keys = jax.random.split(key, num_epochs)
for i in range(num_epochs):
    state, logs = train(jax.random.split(train_keys[i], num_seeds), state, num_steps)

    returned_episode = logs.pop("returned_episode")
    episode_statistics = {
        "episode_returns": logs.pop("returned_episode_returns"),
        "episode_lengths": logs.pop("returned_episode_lengths"),
        "discounted_episode_returns": logs.pop("returned_discounted_episode_returns"),
    }

    data = {}
    if returned_episode.any():
        data |= {
            name: jnp.mean(value, where=returned_episode, axis=(1, 2))
            for name, value in episode_statistics.items()
        }
    print(f"epoch {i + 1}/{num_epochs}: {data}")
