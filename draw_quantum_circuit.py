import pennylane as qml
import random
import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

date = '20211122_120334'

resume_iters = 30

layer = 2

qubits = 8

dev = qml.device('default.qubit', wires=qubits)
@qml.qnode(dev, interface='torch', diff_method='backprop')
def gen_circuit(w):
    z1 = random.uniform(-1, 1)
    z2 = random.uniform(-1, 1)

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

model_dir_path = r'/home/ken/projects/QuantumGAN-PyTorch/results/quantum-GAN/'+date+'/train/model_dir'

weights_pth = os.path.join(model_dir_path, 'molgan_red_weights.csv')

weights = pd.read_csv(weights_pth, header=None).iloc[resume_iters-1, 1:].values

gen_weights = torch.tensor(list(weights), requires_grad=True)

drawer = qml.draw(gen_circuit)

print(drawer(gen_weights))
