import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.nfmodel.utils import *
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
import json
import argparse

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

# Create parser that take a json config file as input
parser = argparse.ArgumentParser(description='Run flowMC on a gaussian mixture model.')
parser.add_argument('--config', type=str, help='Path to the json config file.')
args = parser.parse_args()

# Load the config file
with open(args.config) as f:
    config = json.load(f)

# Extract the parameters from the config file
n_dim = config["n_dim"]
n_chains = config["n_chains"]
n_loop_training = config["n_loop_training"]
n_loop_production = config["n_loop_production"]
n_local_steps = config["n_local_steps"]
n_global_steps = config["n_global_steps"]
learning_rate = config["learning_rate"]
momentum = config["momentum"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]

data = config["data"]
n_block = config["n_blocks"]
hidden_units = config["hidden_units"]
n_bins = config["n_bins"]

output_path = config["output_path"]

data['mu1'] = jnp.array(data['mu1'])
data['mu2'] = jnp.array(data['mu2'])
data['mu3'] = jnp.array(data['mu3'])
data['sigma1'] = jnp.array(data['sigma1'])
data['sigma2'] = jnp.array(data['sigma2'])
data['sigma3'] = jnp.array(data['sigma3'])

# Initialize the random number generator keys
rng_key_set = initialize_rng_keys(n_chains, seed=config["seed"])
initial_position = jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * 5

MALA_Sampler = MALA(mixture_gaussian, True, {"step_size": config["step_size"]})
model = RQSpline(n_dim, n_block, hidden_units, n_bins)

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

nf_sampler.save(output_path)

# # Create a JSON file with the parameters of the model
# with open("example/agent_example/example_config.json", "w") as f:
#     config = {
#         "n_dim": n_dim,
#         "n_chains": n_chains,
#         "n_loop_training": n_loop_training,
#         "n_loop_production": n_loop_production,
#         "n_local_steps": n_local_steps,
#         "n_global_steps": n_global_steps,
#         "learning_rate": learning_rate,
#         "momentum": momentum,
#         "num_epochs": num_epochs,
#         "batch_size": batch_size,
#         "data": data,
#         "seed": 42,
#         "step_size": 1.0,
#         "n_blocks": 4,
#         "hidden_units": [32, 32],
#         "n_bins": 8,
#     }
#     json.dump(config, f, indent=4)