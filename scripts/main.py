# %%
import jax.numpy as jnp
import jax.random as random
from vfim.dynamics import RNNDynamics, GPDynamics, VectorFieldDynamics
from vfim.vector_fim import FIM
from vfim.data_generation import VectorField


if __name__ == "__main__":
    # Step 0: Set up random seed and key
    np_seed = 10
    key = random.PRNGKey(0)

    # Step 1: Generate random vector field and define dynamics model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=np_seed)

    # Step 2: Generate training trajectory (K x T)
    dt = 0.1
    K = 10
    T = 200
    key, subkey = random.split(key)
    x0 = random.normal(subkey, (K, 2))
    true_dynamics = VectorFieldDynamics(dt, vf)
    X_train = true_dynamics.generate_trajectory(x0, n_steps=T)

    # # Step 2: Initialize models and parameters
    # key, subkey = random.split(key)
    # rnn_params = (
    #     random.normal(subkey, (2, 2)),  # W
    #     random.normal(subkey, (2,)),  # U (not used in simple setup)
    #     random.normal(subkey, (2,)),  # b
    # )
    # rnn_model = RNNDynamics(rnn_params)
    # gp_model = GPDynamics(X_train, Y_train)

    # # Step 3: Initialize FIM computation
    # fim_computer_rnn = FIM(rnn_model)
    # fim_computer_gp = FIM(gp_model)

    # # Step 4: Compute FIM and CRLB from two initial points
    # initial_state_1 = jnp.array([1.0, 0.5])
    # initial_state_2 = jnp.array([-1.0, -0.5])

    # fim_rnn_1 = fim_computer_rnn.compute_fim(initial_state_1, rnn_params)
    # fim_rnn_2 = fim_computer_rnn.compute_fim(initial_state_2, rnn_params)

    # print("FIM at initial point 1:")
    # print(fim_rnn_1)
    # print("FIM at initial point 2:")
    # print(fim_rnn_2)

    # crlb_1 = fim_computer_rnn.compute_crlb(fim_rnn_1)
    # crlb_2 = fim_computer_rnn.compute_crlb(fim_rnn_2)

    # print("\nCRLB at initial point 1:")
    # print(crlb_1)
    # print("\nCRLB at initial point 2:")
    # print(crlb_2)

    # # Step 5: MCMC sampling to validate CRLB
    # mcmc_samples_1 = fim_computer_rnn.mcmc_sampling(initial_state_1, rnn_params)
    # mcmc_samples_2 = fim_computer_rnn.mcmc_sampling(initial_state_2, rnn_params)

    # mcmc_cov_1 = jnp.cov(mcmc_samples_1.T)
    # mcmc_cov_2 = jnp.cov(mcmc_samples_2.T)

    # print("\nMCMC covariance at initial point 1:")
    # print(mcmc_cov_1)
    # print("\nMCMC covariance at initial point 2:")
    # print(mcmc_cov_2)

    # # Step 6: Verify CRLB validity
    # validity_1 = jnp.all(jnp.diag(crlb_1) <= jnp.diag(mcmc_cov_1))
    # validity_2 = jnp.all(jnp.diag(crlb_2) <= jnp.diag(mcmc_cov_2))

    # print("\nCRLB Validity at initial point 1:", "Valid" if validity_1 else "Not valid")
    # print("CRLB Validity at initial point 2:", "Valid" if validity_2 else "Not valid")
