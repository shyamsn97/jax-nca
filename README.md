# Neural Cellular Automata (Based on https://distill.pub/2020/growing-ca/) implemented in Jax (Flax)

![Gecko gif](https://raw.githubusercontent.com/shyamsn97/jax-nca/main/images/gecko.gif?token=GHSAT0AAAAAABTB4G7FLAJSLDHSIOQONS3IYTB5ZEA)

---


## Installation
from source:
```
git clone git@github.com:shyamsn97/jax-nca.git
cd jax-nca
python setup.py install
```

from PYPI
```
pip install jax-nca
```
---

## How do NCAs work?
For more information, view the awesome article https://distill.pub/2020/growing-ca/ -- Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020

Image below describes a single update step: https://github.com/distillpub/post--growing-ca/blob/master/public/figures/model.svg

![NCA update](https://raw.githubusercontent.com/shyamsn97/jax-nca/main/images/model.svg?token=GHSAT0AAAAAABTB4G7FOWOPXEUYVLBGRNSWYTB5YUA)

---

## Why Jax?

<b> Note: This project served as a nice introduction to jax, so its performance can probably be improved </b>

NCAs are autoregressive models like RNNs, where new states are calculated from previous ones. With jax, we can make these operations a lot more performant with `jax.lax.scan`  and `jax.jit` (https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)

Instead of writing the nca growth process as:

```python
def multi_step(params, nca, current_state, num_steps):
    # params: parameters for NCA
    # nca: Flax Module describing NCA
    # current_state: Current NCA state
    # num_steps: number of steps to run

    for i in range(num_steps):
        current_state = nca.apply(params, current_state)
    return current_state
```

We can write this with `jax.lax.scan`

```python
def multi_step(params, nca, current_state, num_steps):
    # params: parameters for NCA
    # nca: Flax Module describing NCA
    # current_state: Current NCA state
    # num_steps: number of steps to run

    def forward(carry, inp):
        carry = nca.apply({"params": params}, carry)
        return carry, carry

    final_state, nca_states = jax.lax.scan(forward, current_state, None, length=num_steps)
    return final_state
```
The actual multi_step implementation can be found here: https://github.com/shyamsn97/jax-nca/blob/main/jax_nca/nca.py#L103

---

## Usage
See [notebooks/Gecko.ipynb](notebooks/Gecko.ipynb) for a full example

<b> Currently there's a bug with the stochastic update, so only `cell_fire_rate = 1.0` works at the moment </b>

Creating and using NCA:

```python
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
    ...

from jax_nca.nca import NCA

# usage
nca = NCA(
    num_hidden_channels = 16, 
    num_target_channels = 3,
    trainable_perception = False,
    cell_fire_rate = 1.0,
    alpha_living_threshold = 0.1
)

nca_seed = nca.create_seed(
    nca.num_hidden_channels, nca.num_target_channels, shape=(64,64), batch_size=1
)
rng = jax.random.PRNGKey(0)
params = = nca.init(rng, nca_seed, rng)["params"]
update = nca.apply({"params":params}, nca_seed, jax.random.PRNGKey(10))

# multi step

final_state, nca_states = nca.multi_step(poarams, nca_seed, jax.random.PRNGKey(10), num_steps=32)
```

To train the NCA:
```python
from jax_nca.dataset import ImageDataset
from jax_nca.trainer import EmojiTrainer


dataset = ImageDataset(emoji='🦎', img_size=64)


nca = NCA(
    num_hidden_channels = 16, 
    num_target_channels = 3,
    trainable_perception = False,
    cell_fire_rate = 1.0,
    alpha_living_threshold = 0.1
)

trainer = EmojiTrainer(dataset, nca, n_damage=0)

trainer.train(100000, batch_size=8, seed=10, lr=2e-4, min_steps=64, max_steps=96)

# to access train state:

state = trainer.state

# save
nca.save(state.params, "saved_params")

# load params
loaded_params = nca.load("saved_params")

```