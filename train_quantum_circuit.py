#!/usr/bin/env python


# training quantum circuit to generate the gaussian distribution

import pennylane as qml
import numpy as np
import random
import os
import pandas as pd
import torch
import csv

model_dir = 'results/quantum_circuit/'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

np.random.seed(42)

qubits = 8

layer = 1

batch_size = 512

iteration = 30000

z_dim = 8

lr = 0.04

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_z(sbatch_size):
    """Sample the random noise"""
    return np.random.normal(0, 1, size=(batch_size, z_dim))

dev = qml.device('default.qubit', wires=qubits)
@qml.qnode(dev, interface='torch', diff_method='backprop')
def gen_circuit(w):
    # random noise as generator input
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)
    # construct generator circuit for both atom vector and node matrix
    for i in range(qubits):
        qml.RY(np.arcsin(z1), wires=i)
        qml.RZ(np.arcsin(z2), wires=i)
    for l in range(layer):
        for i in range(qubits):
            qml.RY(w[i], wires=i)
        for i in range(qubits-1):
            qml.CNOT(wires=[i, i+1])
            qml.RZ(w[i+qubits], wires=i+1)
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(qubits)]

gen_weights = torch.tensor(list(np.random.rand(layer*(qubits*2-1))*2*np.pi-np.pi), requires_grad=True)

best_params = torch.tensor(list(np.random.rand(layer*(qubits*2-1))*2*np.pi-np.pi), requires_grad=True)

opt = torch.optim.RMSprop([gen_weights], lr)

best_cost = 2.0

for n in range(iteration):
    opt.zero_grad()
    loss = torch.nn.MSELoss()
    z = sample_z(batch_size)
    z1 = torch.from_numpy(z).to(device).float()
    sample_list = [gen_circuit(gen_weights) for i in range(batch_size)]
    z2 = torch.stack(tuple(sample_list)).to(device).float()
    output = loss(z1, z2)
    output.backward()
    opt.step()

    if output < best_cost:
        best_cost = output
        best_params = gen_weights
        print(n, best_cost)
        with open(os.path.join(model_dir, 'molgan_red_weights.csv'), 'a') as file:
            writer = csv.writer(file)
            writer.writerow([str(n)]+list(best_params.detach().numpy()))

