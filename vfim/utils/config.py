from dataclasses import dataclass, asdict

from matplotlib.artist import get


@dataclass(match_args=False)
class Config:
    d_obs: int = 50
    d_latent: int = 2
    d_hidden: int = 16

    # For training
    batch_size: int = 64
    n_epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # For active learning
    num_trials: int = 50
    trial_length: int = 5
    fisher_length: int = 5
    n_epoch_refine: int = 10
    lr_refine: float = 1e-4
    u_strength: float = 0.15

    @classmethod
    def exp0(cls):
        return cls(d_obs=50, d_latent=2, d_hidden=16)

    def dict_(self):
        return asdict(self)


def get_config(exp_name):
    return getattr(Config, exp_name)()
