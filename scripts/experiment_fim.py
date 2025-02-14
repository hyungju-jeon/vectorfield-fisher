# %%
import torch
import torch.nn as nn
from torch import autograd, optim
import matplotlib.pyplot as plt
import numpy as np
from vfim.dynamics import RNNDynamics, GPDynamics, VectorFieldDynamics
from vfim.fisher_information import FIM
from vfim.vector_field import VectorField


if __name__ == "__main__":
    # Step 0: Set up random seed and key
    np_seed = 10

    # Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=np_seed)
    vf.streamplot()

    dt = 0.1
    true_dynamics = VectorFieldDynamics(dt, vf)

    # # Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    # K = 20
    # T = 200
    # key, subkey = random.split(key)
    # x0 = random.normal(key, (K, 2))

    # x_train = true_dynamics.generate_trajectory(x0, T)

    # Step 3: Initialize the inference model and train
    # Step 4: Compute FIM and CRLB from two initial points
    # Step 5: Update the inference model and evaluate the model accuracy
