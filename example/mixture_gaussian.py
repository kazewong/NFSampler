import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.nfmodel.utils import *
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys


def mixture_gaussian(x: jnp.array, params: dict) -> float:
    """Probability density function of a gaussian mixture model with 3 Gaussians.
    Attribute
    ---------
    x: jnp.ndarray (n_dim,)
        Position at which to evaluate the probability density function.
    params: dict
        Parameters of the gaussian mixture model.
        The keys of the parameters should be mu1, mu2, mu3, sigma1, sigma2, sigma3, w1, w2, w3.

    Returns
    -------
    float
        Log-probability density function of the gaussian mixture model at position x.
    """
    
    mu1 = params["mu1"]
    mu2 = params["mu2"]
    mu3 = params["mu3"]
    sigma1 = params["sigma1"]
    sigma2 = params["sigma2"]
    sigma3 = params["sigma3"]
    w1 = params["w1"]
    w2 = params["w2"]
    w3 = params["w3"]
    
    logpdf1 = multivariate_normal.logpdf(x, mu1, sigma1)
    logpdf2 = multivariate_normal.logpdf(x, mu2, sigma2)
    logpdf3 = multivariate_normal.logpdf(x, mu3, sigma3)

    logpdf = jnp.log(w1 * jnp.exp(logpdf1) + w2 * jnp.exp(logpdf2) + w3 * jnp.exp(logpdf3))
    
    return logpdf

n_dim = 2
n_chains = 20
n_loop_training = 10
n_loop_production = 10
n_local_steps = 100
n_global_steps = 100
learning_rate = 0.001
momentum = 0.9
num_epochs = 30
batch_size = 10000

data = {"mu1": jnp.array([0.0, 0.0]), "mu2": jnp.array([3.0, 3.0]), "mu3": jnp.array([4.0, -3.0]),
        "sigma1": jnp.array([[1.0, 0.0], [0.0, 1.0]]), "sigma2": jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        "sigma3": jnp.array([[1.0, 0.0], [0.0, 1.0]]), "w1": 0.2, "w2": 0.5, "w3": 0.3}

rng_key_set = initialize_rng_keys(n_chains, seed=42)

initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 1

MALA_Sampler = MALA(mixture_gaussian, True, {"step_size": 1.0})
model = RQSpline(n_dim, 4, [32, 32], 8)

print("Initializing sampler class")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    data,
    MALA_Sampler,
    model,
    n_loop_training=n_loop_training,
    n_loop_production=n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
)

nf_sampler.sample(initial_position, data)
summary = nf_sampler.get_sampler_state(training=True)
chains, log_prob, local_accs, global_accs, loss_vals = summary.values() 
nf_samples = nf_sampler.sample_flow(10000)

nf_sampler.save("test_workspace/mixture_gaussian")
