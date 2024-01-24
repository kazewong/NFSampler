from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
import jax
import jax.numpy as jnp  # JAX NumPy

from flowMC.nfmodel.utils import *
import equinox as eqx
import optax  # Optimizers
from flowMC.nfmodel.utils import make_training_loop

from sklearn.datasets import make_moons

"""
Training a Masked Coupling RQSpline flow to fit the dual moons dataset.
"""

num_epochs = 3000
batch_size = 10000
learning_rate = 0.001
momentum = 0.9
n_layers = 10
n_hidden = 128
dt = 1 / n_layers

data = make_moons(n_samples=20000, noise=0.05)
data = jnp.array(data[0])

key1, rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)

def create_model_and_state(rng, learning_rate):
    model = MaskedCouplingRQSpline(
    2,
    4,
    [32, 32],
    8,
    rng,
    data_cov=jnp.cov(data.T),
    data_mean=jnp.mean(data, axis=0),
)
    optim = optax.adam(learning_rate)
    state = optim.init(eqx.filter(model, eqx.is_array))
    return model, state

def loss_fn(model: MaskedCouplingRQSpline, data):
    return -jnp.mean(model.log_prob(data))

def train_step(model, state, data, rng):
    loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, data)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss
