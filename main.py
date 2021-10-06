import os
import logging

from rdkit import RDLogger

from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix

# Remove flooding logs.
#lg = RDLogger.logger()
#lg.setLevel(RDLogger.CRITICAL)

from solver import Solver
from torch.backends import cudnn

import pennylane as qml
import random
import numpy as np

def main(config):

    # For fast training
    cudnn.benchmark = True

    # Timestamp
    if config.mode == 'train':
        a_train_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir, a_train_time)
        config.log_dir_path = os.path.join(config.saving_dir, config.mode, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, config.mode, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, config.mode, 'img_dir')
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'log_dir')
        config.model_dir_path = os.path.join(config.saving_dir, 'model_dir')
        config.img_dir_path = os.path.join(config.saving_dir, 'post_test', a_test_time, 'img_dir')


    # Create directories if not exist
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == 'train':
        log_p_name = os.path.join(config.log_dir_path, a_train_time + '_logger.log')
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)


    exit()

    # Solver for training and test MolGAN
    if config.mode == 'train':
        solver = Solver(config, logging)
    elif config.mode == 'test':
        solver = Solver(config)
    else:
        raise NotImplementedError

    solver.train_and_validate()

if __name__ == '__main__':

    config = get_GAN_config()

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="6"

    # molecule dataset dir
    config.mol_data_dir = r'data/gdb9_9nodes.sparsedataset'
    #config.mol_data_dir = r'data/qm9_5k.sparsedataset'

    # Quantum
    config.quantum = True
    config.complexity = 'nr'
    config.batch_size = 128

    # Training
    config.z_dim = 8
    config.num_epochs = 30
    # 1.0 for pure WGAN and 0.0 for pure RL
    config.lambda_wgan = 1.0

    # Test
    #config.mode = "test"
    #config.test_epoch = 30
    #config.test_sample_size = 5000
    #config.z_dim = 32
    #config.saving_dir = r"results/GAN/20210929_175628/train"



    # Quantum
    if config.complexity == 'nr':
        config.g_conv_dim = [128, 256, 512]
    elif config.complexity == 'mr':
        config.g_conv_dim = [128]
    elif config.complexity == 'hr':
        config.g_conv_dim = [16]
    else:
        raise ValueError("Please enter an valid model complexity from 'mr', 'hr' or 'nr'!")

    if config.quantum:
        config.saving_dir = 'results/quantum-GAN'

    dev = qml.device('default.qubit', wires=config.qubits)
    @qml.qnode(dev, interface='torch')
    def gen_circuit(w):
        # random noise as generator input
        z1 = random.uniform(-1, 1)
        z2 = random.uniform(-1, 1)

        # construct generator circuit for both atom vector and node matrix
        for i in range(config.qubits):
            qml.RY(np.arcsin(z1), wires=i)
            qml.RZ(np.arcsin(z2), wires=i)
        for l in range(config.layer):
            for i in range(config.qubits):
                qml.RY(w[i], wires=i)
            for i in range(config.qubits-1):
                qml.CNOT(wires=[i, i+1])
                qml.RZ(w[i+config.qubits], wires=i+1)
                qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(config.qubits)]

    config.gen_circuit = gen_circuit

    print(config)

    main(config)
