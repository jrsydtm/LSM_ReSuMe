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


# Function to calculate Euclidean distance between spike trains
def euclidean_distance(spike_train1, spike_train2):
    # Ensure both spike trains have the same length by padding with zeros
    length = max(len(spike_train1), len(spike_train2))
    spike_train1 = np.pad(spike_train1, (0, length - len(spike_train1)), "constant")
    spike_train2 = np.pad(spike_train2, (0, length - len(spike_train2)), "constant")
    return np.sqrt(np.sum((spike_train1 - spike_train2) ** 2))


# Define basic parameters
np.random.seed(42)
N_input = X_train.shape[1]
N_liquid = 2000
N_output = len(np.unique(y_train))
single_example_time = 1 * second

# Neuron model parameters
Ee = 0.0 * mV
El = -74.0 * mV
v_reset = -60.0 * mV
v_thresh = -54.0 * mV
taum = 100.0 * ms
taue = 50.0 * ms

# STDP parameters
taupre = 20.0 * ms
taupost = 20.0 * ms
gmax = 0.01
dApre = 0.01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# ReSuMe parameters
taul = 20.0 * ms
taud = 20.0 * ms
dAd = 0.01
dAl = 0.01
dAl *= gmax
dAd *= gmax

# Define liquid neuron equations
eqs_neurons_liquid = """
dv/dt = (ge * (Ee - v) + El - v) / taum : volt
dge/dt = -ge / taue : 1
"""

# Create input groups
input_groups = PoissonGroup(N_input, 0 * Hz)
# Create teacher groups
teacher_groups = SpikeGeneratorGroup(N_output, [], [] * ms)
# Define liquid neuron groups
neurons_liquid = NeuronGroup(
    N_liquid,
    eqs_neurons_liquid,
    threshold="v > v_thresh",
    reset="v = v_reset",
    method="euler",
)
# Define output neuron groups
neurons_output = NeuronGroup(
    N_output,
    eqs_neurons_output,
    threshold="v > v_thresh",
    reset="v = v_reset",
    method="euler",
)

# Define synapses between input layer and liquid layer
S_input_to_liquid = Synapses(
    input_groups,
    neurons_liquid,
    """
    w : 1
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
    """,
    on_pre="""ge += w
              Apre += dApre
              w = clip(w + Apost, 0, gmax)""",
    on_post="""Apost += dApost
               w = clip(w + Apre, 0, gmax)""",
)
S_input_to_liquid.connect()
S_input_to_liquid.w = "rand() * gmax"

# Define synapses within the liquid layer based on distance
S_liquid_internal = Synapses(
    neurons_liquid, neurons_liquid, "w : 1", on_pre="ge_post += w"
)
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
S_liquid_to_output = Synapses(
    neurons_liquid,
    neurons_output,
    """
    w : 1
    dAl/dt = -Al / taul : 1 (event-driven)
    dAd/dt = -Ad / taud : 1 (event-driven)
    """,
    on_pre="""ge_post += w""",
    on_post="""Al += dAl; w = clip(w - Al, 0, gmax)""",
)
S_liquid_to_output.connect()
S_liquid_to_output.w = "rand() * gmax"

# Define synapses between liquid layer and teacher layer
S_liquid_to_teachers = Synapses(
    neurons_liquid,
    teacher_groups,
    " ",
    on_post="""Ad += dAd; w = clip(w + Ad, 0, gmax)""",
)
S_liquid_to_teachers.connect()
S_liquid_to_teachers.variables.add_reference("w", S_liquid_to_output, "w")
S_liquid_to_teachers.variables.add_reference("Al", S_liquid_to_output, "Al")
S_liquid_to_teachers.variables.add_reference("Ad", S_liquid_to_output, "Ad")

# Create monitors
spikemon_teacher = SpikeMonitor(teacher_groups)
spikemon_output = SpikeMonitor(neurons_output)

state_monitor_output = StateMonitor(neurons_output, True, record=True, dt=1 * second)
state_monitor_liquid_to_output = StateMonitor(
    S_liquid_to_output, "w", record=True, dt=1 * second
)

# Lists to store Euclidean distances for each output neuron
distances = {i: [] for i in range(N_output)}
iterations = []

# Define desired spike trains
desired_spikes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900] * ms

# Run multiple iterations of training
num_iterations = 1000
input_groups.rates = 0 * Hz
run(0 * second)

for iteration in range(num_iterations):
    input_spike_rates = X_train.iloc[0, :].values.astype(float) * 108.0
    input_groups.rates = input_spike_rates * Hz

    teacher = y_train.iloc[0].astype(int)
    teacher_spikes = desired_spikes + 1200 * iteration * ms
    teacher_groups.set_spikes([teacher] * len(teacher_spikes), teacher_spikes)

    run(single_example_time, report="text")

    # Calculate Euclidean distance
    for k in range(N_output):
        output_spike_trains = np.array(
            spikemon_output.spike_trains()[k] / ms
        )  # Convert to ms and remove units
        desired_spike_trains = desired_spikes / ms  # Convert to ms and remove units
        # Normalize spike times so that each sequence starts from 0
        if len(output_spike_trains) > 0:
            output_spike_trains -= output_spike_trains[0]
        output_spike_trains -= iteration * 1200
        actual_output_spike_trains = output_spike_trains[output_spike_trains > 0]
        print("output spike trains:", actual_output_spike_trains)
        print("desired spike trains", desired_spike_trains)
        distance = euclidean_distance(actual_output_spike_trains, desired_spike_trains)
        distances[k].append(distance)
    iterations.append(iteration)

# Save weights
np.save("ReSuMe_weights_input_to_liquid.npy", S_input_to_liquid.w)
np.save("ReSuMe_weights_liquid_internal.npy", S_liquid_internal.w)
np.save("ReSuMe_weights_liquid_to_output.npy", S_liquid_to_output.w)
# Save connections
np.save(
    "ReSuMe_connections_input_to_liquid.npy",
    np.vstack((S_input_to_liquid.i, S_input_to_liquid.j)).T,
)
np.save(
    "ReSuMe_connections_liquid_internal.npy",
    np.vstack((S_liquid_internal.i, S_liquid_internal.j)).T,
)
np.save(
    "ReSuMe_connections_liquid_to_output.npy",
    np.vstack((S_liquid_to_output.i, S_liquid_to_output.j)).T,
)
print("Training complete and weights and connections saved.")

# # Retrieve and print spike times
# print("Spike times for output neurons:")
# for i in range(N_output):
#     print("Neuron", i, "spikes at times:", spikemon_output.spike_trains()[i])
# print("Spike times for teacher neurons:")
# for i in range(N_output):
#     print("Neuron", i, "spikes at times:", spikemon_teacher.spike_trains()[i])

# Plot spike trains
# figure()
# for i in range(N_output):
#     plot(spikemon_teacher.t / ms, spikemon_teacher.i, ".k")
# xlabel("Time (ms)")
# ylabel("Neuron index")
# title("Spikes of Teacher Neurons")
# show()

figure()
for i in range(N_output):
    plot(state_monitor_output.t / ms, state_monitor_output.v[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("Voltage (V)")
title("Voltage Change of Output Neuron")
legend()
show()

figure()
for i in range(N_output):
    plot(state_monitor_output.t / ms, state_monitor_output.ge[i], label=f"Neuron {i}")
xlabel("Time (ms)")
ylabel("ge")
title("ge of Output Neuron")
legend()
show()

# Plot synaptic connectivity and weights
fig, axs = plt.subplots(6, 1, figsize=(12, 24))
axs[0].plot(S_input_to_liquid.w, ".k")
axs[0].set_ylabel("Weight")
axs[0].set_xlabel("Synapse index")
axs[0].set_title("Synaptic Connectivity: Input to Liquid Layer")
axs[1].hist(S_input_to_liquid.w, 20)
axs[1].set_xlabel("Weight")
axs[1].set_title("Weight Distribution: Input to Liquid Layer")
axs[2].plot(S_liquid_internal.w, ".k")
axs[2].set_ylabel("Weight")
axs[2].set_xlabel("Synapse index")
axs[2].set_title("Synaptic Connectivity: Liquid Internal Layer")
axs[3].hist(S_liquid_internal.w, 20)
axs[3].set_xlabel("Weight")
axs[3].set_title("Weight Distribution: Liquid Internal Layer")
axs[4].plot(S_liquid_to_output.w, ".k")
axs[4].set_ylabel("Weight")
axs[4].set_xlabel("Synapse index")
axs[4].set_title("Synaptic Connectivity: Liquid to Output Layer")
axs[5].hist(S_liquid_to_output.w, 20)
axs[5].set_xlabel("Weight")
axs[5].set_title("Weight Distribution: Liquid to Output Layer")
plt.tight_layout()
plt.show()

# Plot weight changes over time
plot(state_monitor_liquid_to_output.t / second, state_monitor_liquid_to_output.w.T)
xlabel("Time (s)")
ylabel("Weight")
title("Synaptic Connection Weights: Liquid to Output Layer")
show()

# Plot Euclidean distances over iterations
figure()
for k in range(N_output):
    plot(iterations, distances[k], label=f"Neuron {k}")
xlabel("Iteration")
ylabel("Euclidean Distance")
title("Euclidean Distance Between Output and Teacher Spike Trains Over Iterations")
legend()
show()

# # Plot synaptic connectivity heatmaps
# weights_input_to_liquid = np.reshape(S_input_to_liquid.w, (N_liquid, N_input))
# weights_liquid_internal = np.reshape(S_liquid_internal.w, (N_liquid, N_liquid))
# weights_liquid_to_output = np.reshape(S_liquid_to_output.w, (N_output, N_liquid))

# fig, axs = plt.subplots(1, 3, figsize=(12, 6))
# cax0 = axs[0].imshow(weights_input_to_liquid, cmap="viridis", origin="lower")
# axs[0].set_xlabel("Source Neuron Index")
# axs[0].set_ylabel("Target Neuron Index")
# axs[0].set_title("Synaptic Connectivity: Input to Liquid Layer")
# plt.colorbar(cax0, ax=axs[0], label="Connection Weight")

# cax1 = axs[1].imshow(weights_liquid_internal, cmap="viridis", origin="lower")
# axs[1].set_xlabel("Source Neuron Index")
# axs[1].set_ylabel("Target Neuron Index")
# axs[1].set_title("Synaptic Connectivity: Liquid Layer to Liquid Layer")
# plt.colorbar(cax1, ax=axs[1], label="Connection Weight")

# cax2 = axs[2].imshow(weights_liquid_to_output, cmap="viridis", origin="lower")
# axs[2].set_xlabel("Source Neuron Index")
# axs[2].set_ylabel("Target Neuron Index")
# axs[2].set_title("Synaptic Connectivity: Liquid Layer to Output Layer")
# plt.colorbar(cax2, ax=axs[2], label="Connection Weight")

# plt.tight_layout()
# plt.show()
