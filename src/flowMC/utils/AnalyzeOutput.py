import argparse
import numpy as np
import corner
import matplotlib.pyplot as plt


# Create a parser which load a numpy binary file, allow different modes for differnt output
parser = argparse.ArgumentParser(description='Analyzing the output of flowMC.')
parser.add_argument('--file', type=str, help='Path to the numpy binary file.')
parser.add_argument('--mode', type=str, help='Mode of the output.', choices=['global_acceptance', 
                                                                             'local_acceptance', 
                                                                             'plot_corner',
                                                                             'plot_log_prob'])
parser.add_argument('--training', type=bool, help='Whether the output is from training.')

args = parser.parse_args()

# Load the numpy binary file
samples = np.load(args.file, allow_pickle=True)

# Extract the samples
if args.training:
    samples = samples["training"].tolist()
else:
    samples = samples["production"].tolist()

chains = np.array(samples["chains"])
log_probs = np.array(samples["log_prob"])
local_accs = np.array(samples["local_accs"])
global_accs = np.array(samples["global_accs"])

output_tag = args.file.replace(".npz", "")

if args.mode == "global_acceptance":
    print("Global acceptance rate: ", np.mean(global_accs))
elif args.mode == "local_acceptance":
    print("Local acceptance rate: ", np.mean(local_accs))
elif args.mode == "plot_corner":
    # Plot the corner plot
    fig = corner.corner(chains.reshape(-1, chains.shape[-1]), 
                        labels=[f"$x_{i}$" for i in range(chains.shape[-1])],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_kwargs={"fontsize": 12})
    fig.savefig(output_tag + "_corner.png")

elif args.mode == "plot_log_prob":
    # Plot the log prob
    plt.plot(log_probs)
    plt.savefig(output_tag + "_log_prob.png")