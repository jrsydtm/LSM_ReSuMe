"""
Filename: ReSuMe.py
Author: Yi Wan
Date: 2024-06-14
Description: Implementation of ReSuMe rule between liquid layer and output layer.
"""

from brian2 import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize the simulation environment
start_scope()

# Read data from an Excel file
df = pd.read_excel("dataset.xlsx", header=None)

# Segment features and labels
X_data = df.iloc[:, :34]
y_data = df.iloc[:, 34]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


# Function to calculate distance-based connection probability
def distance_probability(pos1, pos2, sigma=0.1):
    distance = np.linalg.norm(pos1 - pos2)
    return np.exp(-(distance**2) / (2 * sigma**2))


# Function to calculate Euclidean distance between two spike trains
def euclidean_distance(spike_train1, spike_train2):
    # Ensure both spike trains have the same length by padding with zeros
    length = max(len(spike_train1), len(spike_train2))
    spike_train1 = np.pad(spike_train1, (0, length - len(spike_train1)), "constant")
    spike_train2 = np.pad(spike_train2, (0, length - len(spike_train2)), "constant")
    return np.sqrt(np.sum((spike_train1 - spike_train2) ** 2))


# Function to save weights
def save_weights():
    return {
        "S_input_liquid": S_input_liquid.w[:].copy(),
        "S_liquid_internal": S_liquid_internal.w[:].copy(),
        "S_liquid_output": S_liquid_output.w[:].copy(),
    }


# Function to load weights
def load_weights(weights):
    S_input_liquid.w[:] = weights["S_input_liquid"]
    S_liquid_internal.w[:] = weights["S_liquid_internal"]
    S_liquid_output.w[:] = weights["S_liquid_output"]


# Plot weight changes over iterations
def plot_weight_changes(weights, title):
    weights = np.array(weights)
    plt.figure(figsize=(12, 6))
    plt.plot(weights)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.show()


# Function to compute the exponential decay kernel Wd for the ReSuMe learning rule
def resume_kernel_d(sd):
    Ad = 0.1  # Learning rate constant
    taud = 20  # Learning time constant
    if sd > 0:
        return Ad * np.exp(-sd / taud)
    else:
        return 0


# Function to compute the exponential decay kernel Wl for the ReSuMe learning rule
def resume_kernel_l(sl):
    Al = 0.1  # Learning rate constant
    taul = 20  # Learning time constant
    if sl > 0:
        return -Al * np.exp(-sl / taul)
    else:
        return 0
    

# Function to update synaptic weights from liquid layer to output layer using the ReSuMe learning rule
def resume_output_weights():
    ad = 0.0001  # non-Hebbian weight term(desired spike trains)
    al = 0.0001  # non-Hebbian weight term(actual spike trains)

    # Initialize weight changes
    dw = np.zeros((N_output, N_liquid))

    # Update weights using ReSuMe rule
    for j in range(N_output):  # Loop over output neurons
        Sl = np.array(spikemon_output.spike_trains()[j] / ms)
        print(f"Output Neuron{j} Sl:",Sl)
        Sd = np.unique(desired_spike_trains[j] / ms)
        print(f"Desired Neuron{j} Sd:",Sd)

        for i in range(N_liquid):  # Loop over liquid neurons
            Sin = np.array(spikemon_liquid.spike_trains()[i] / ms)
            dw_tmp = 0

            # desired output contribution to the weight change
            for td in Sd[-5:]:
                dw_tmp += ad
                for sd in Sin - td:
                    dw_tmp += resume_kernel_d(sd)

            # actual output contribution to the weight change
            for tl in Sl[-5:]:
                dw_tmp -= al
                for sl in Sin - tl:
                    dw_tmp += resume_kernel_l(sl)
        
            # Update the weight change
            dw[j, i] = dw_tmp
    
    return dw / float(N_liquid)


# Define basic parameters
np.random.seed(42)
N_input = X_train.shape[1]
N_liquid = 2000
N_output = len(np.unique(y_train))
single_example_time = 1 * second
num_iterations = 100
run_time = num_iterations * single_example_time

# Neuron model parameters
Ee = 0.0 * mV
El = -74.0 * mV
vr = -60.0 * mV
vt = -54.0 * mV
taum = 100.0 * ms
taue = 65.0 * ms

# STDP parameters
taupre = 20.0 * ms
taupost = 20.0 * ms
gmax = 0.01
dApre = 0.01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# Define neuron equations
eqs_liquid = """
dv/dt = (ge * (Ee - v) + El - v)/taum : volt
dge/dt = -ge/taue : 1
"""
eqs_output = """
dv/dt = (ge * (Ee - v) + El - v)/(50 * ms) : volt
dge/dt = -ge/(1 * ms) : 1
"""

# Create input groups
input_neurons = PoissonGroup(N_input, 0 * Hz)
# Create teacher groups
teacher_neurons = SpikeGeneratorGroup(N_output, [], [] * ms)
# Define liquid neuron groups
liquid_neurons = NeuronGroup(N_liquid, eqs_liquid, threshold="v > vt", reset="v = vr", method="euler")
# Define output neuron groups
output_neurons = NeuronGroup(N_output, eqs_output, threshold="v > vt", reset="v = vr", method="euler")

# Define synapses between input layer and liquid layer
S_input_liquid = Synapses(input_neurons, liquid_neurons, "w : 1", on_pre="ge += w")
S_input_liquid.connect()
S_input_liquid.w = "rand() * gmax"

# Define synapses within the liquid layer based on distance
S_liquid_internal = Synapses(liquid_neurons, liquid_neurons, "w : 1", on_pre="ge_post += w")
# Assign positions to liquid neurons in 3D
positions = np.random.rand(N_liquid, 3)  # Uniformly distributed positions in a 3D space
# Connect based on distance probability in 3D
for i in range(N_liquid):
    for j in range(N_liquid):
        if i != j:
            prob = distance_probability(positions[i], positions[j])
            if np.random.rand() < prob:
                S_liquid_internal.connect(i=i, j=j)
S_liquid_internal.w = "rand() * gmax"

# Define synapses between liquid layer and output layer
S_liquid_output = Synapses(liquid_neurons, output_neurons, "w : 1", on_pre="ge_post += w")
S_liquid_output.connect()
S_liquid_output.w = "rand() * gmax"

# Create monitors
spikemon_input = SpikeMonitor(input_neurons)
spikemon_teacher = SpikeMonitor(teacher_neurons)
spikemon_liquid = SpikeMonitor(liquid_neurons)
spikemon_output = SpikeMonitor(output_neurons)

statemon_liquid = StateMonitor(liquid_neurons, ['v', 'ge'], record=True, dt=1 * ms)
statemon_output = StateMonitor(output_neurons, ['v', 'ge'], record=True, dt=1 * ms)

# statemon_input_liquid = StateMonitor(S_liquid_internal, "w", record=True)
# statemon_liquid_output = StateMonitor(S_liquid_output, "w", record=True)

# Define desired spike trains
desired_spikes = [750, 800, 850, 900, 950] * ms

# Store the initial state of the network
store()

# Lists to store weights over iterations
input_liquid_weights = []
liquid_internal_weights = []
liquid_output_weights = []

# Record the initial weights
input_liquid_weights.append(S_input_liquid.w[:].copy())
liquid_internal_weights.append(S_liquid_internal.w[:].copy())
liquid_output_weights.append(S_liquid_output.w[:].copy())

# Lists to store Euclidean distances for each output neuron
distances = {i: [] for i in range(N_output)}
iterations = []

for iteration in range(num_iterations):
    print(f"Iteration {iteration + 1}/{num_iterations}")
    # Resotre the original state of the network
    restore()

    # Assign values to the spike rates of input neurons
    input_spike_rates = X_train.iloc[0, :].values.astype(float) * 108.0
    input_neurons.rates = input_spike_rates * Hz

    # Assign desired spike trains of teacher neurons
    teacher = y_train.iloc[0].astype(int)
    teacher_neurons.set_spikes([teacher] * len(desired_spikes), desired_spikes)
    desired_spike_trains = np.zeros((N_output, len(desired_spikes))) * ms
    desired_spike_trains[teacher,:] = desired_spikes

    # Load weights from previous iteration
    if iteration > 0:
        load_weights(saved_weights)

    # Run the simulation
    run(single_example_time, report="text")

    # Update weights using ReSuMe
    dw = resume_output_weights()
    dw_flattened = dw.flatten()
    # print("dw_flattened shape:", dw_flattened.shape)
    # # Ensure shapes match
    # assert S_liquid_output.w[:].shape == dw_flattened.shape, "Shapes do not match after flattening dw"
    S_liquid_output.w[:] += dw_flattened
    S_liquid_output.w[:] = np.clip(S_liquid_output.w[:], 0, gmax)  # Limit the weights to [0, gmax]

    # Record the weights
    input_liquid_weights.append(S_input_liquid.w[:].copy())
    liquid_internal_weights.append(S_liquid_internal.w[:].copy())
    liquid_output_weights.append(S_liquid_output.w[:].copy())

    # Save current weights
    saved_weights = save_weights()

    # Calculate Euclidean distance
    for k in range(N_output):
        output_spike_trains = np.array(spikemon_output.spike_trains()[k] / ms)  # Remove units
        desired_single_spike_trains = desired_spikes / ms  # Remove units
        
        distance = euclidean_distance(output_spike_trains[:5], desired_single_spike_trains[:5])
        distances[k].append(distance)

        # Debugging info
        print(f"Output neuron{k}:")
        print("output spike trains:", output_spike_trains)
        print("desired spike trains", desired_single_spike_trains)
        print(distance)
    
    iterations.append(iteration)

# Plot voltage change of liquid neurons
figure()
for i in range(N_liquid):
    plot(statemon_liquid.t / ms, statemon_liquid.v[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("Voltage (mV)")
title("Voltage Change of Liquid Neuron")
legend()
show()
# Plot ge change of liquid neurons
figure()
for i in range(N_liquid):
    plot(statemon_liquid.t / ms, statemon_liquid.ge[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("ge")
title("ge of Liquid Neuron")
legend()
show()
# Plot voltage change of output neurons
figure()
for i in range(N_output):
    plot(statemon_output.t / ms, statemon_output.v[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("Voltage (mV)")
title("Voltage Change of Output Neuron")
legend()
show()
# Plot ge change of output neurons
figure()
for i in range(N_output):
    plot(statemon_output.t / ms, statemon_output.ge[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("ge")
title("ge of Output Neuron")
legend()
show()

# Plot synaptic connectivity and weights
fig, axs = plt.subplots(6, 1, figsize=(12, 24))
axs[0].plot(S_input_liquid.w, ".k")
axs[0].set_ylabel("Weight")
axs[0].set_xlabel("Synapse index")
axs[0].set_title("Synaptic Connectivity: Input to Liquid Layer")
axs[1].hist(S_input_liquid.w, 20)
axs[1].set_xlabel("Weight")
axs[1].set_title("Weight Distribution: Input to Liquid Layer")
axs[2].plot(S_liquid_internal.w, ".k")
axs[2].set_ylabel("Weight")
axs[2].set_xlabel("Synapse index")
axs[2].set_title("Synaptic Connectivity: Liquid Internal Layer")
axs[3].hist(S_liquid_internal.w, 20)
axs[3].set_xlabel("Weight")
axs[3].set_title("Weight Distribution: Liquid Internal Layer")
axs[4].plot(S_liquid_output.w, ".k")
axs[4].set_ylabel("Weight")
axs[4].set_xlabel("Synapse index")
axs[4].set_title("Synaptic Connectivity: Liquid to Output Layer")
axs[5].hist(S_liquid_output.w, 20)
axs[5].set_xlabel("Weight")
axs[5].set_title("Weight Distribution: Liquid to Output Layer")
plt.tight_layout()
plt.show()

# Plot the weight changes
plot_weight_changes(liquid_output_weights, "Liquid to Output Layer Weights Over Iterations")

# Plot Euclidean distances over iterations
figure()
for k in range(N_output):
    plot(iterations, distances[k], label=f"Neuron {k}")
xlabel("Iteration")
ylabel("Euclidean Distance")
title("Euclidean Distance Between Output and Teacher Spike Trains Over Iterations")
legend()
show()
