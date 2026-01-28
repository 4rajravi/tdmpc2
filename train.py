import os
import hydra
from omegaconf import DictConfig

from tdmpc2.utils.seed import set_seed
from tdmpc2.utils.torch_utils import get_device

from tdmpc2.envs.make_env import make_env

from tdmpc2.models.encoder.encoder_factory import make_encoder
from tdmpc2.models.simnorm import SimNorm
from tdmpc2.models.world_model import WorldModel
from tdmpc2.models.policy_prior.gaussian_policy import GaussianPolicy

from tdmpc2.planning.mppi import MPPI
from tdmpc2.planning.mpc import MPC

from tdmpc2.training.trainer import Trainer
from tdmpc2.evaluation.evaluator import Evaluator
from tdmpc2.evaluation.csv_logger import CSVLogger

import warnings
warnings.filterwarnings("ignore")


@hydra.main(
    version_base="1.2",
    config_path="configs",
    config_name="config"
)
def main(cfg: DictConfig):

    # setup
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = get_device()
    set_seed(cfg.seed)

    print(f"Device: {device}")
    print(f"Task: {cfg.env.task}")

    # environments
    env = make_env(cfg.env)
    eval_env = make_env(cfg.env)

    obs, _ = env.reset()
    obs_shape = obs.shape

    csv_logger = CSVLogger(
    os.path.join(cfg.work_dir, "eval.csv"))

    action_dim = (
        env.action_space.n
        if cfg.env.discrete_actions
        else env.action_space.shape[0]
    )

    # encoder
    encoder = make_encoder(
        cfg.obs,
        cfg.model,
        obs_shape
    ).to(device)

    simnorm = SimNorm(
        dim=cfg.model.latent_dim,
        groups=cfg.model.simnorm_dim
    ).to(device)

    # world model
    world_model = WorldModel(
        latent_dim=cfg.model.latent_dim,
        action_dim=action_dim,
        mlp_dim=cfg.model.mlp_dim,
        num_q=cfg.model.num_q,
        num_bins=cfg.model.num_bins,
        dropout=cfg.model.dropout
    ).to(device)

    # policy prior
    policy_prior = GaussianPolicy(
        latent_dim=cfg.model.latent_dim,
        action_dim=action_dim,
        log_std_min=cfg.model.log_std_min,
        log_std_max=cfg.model.log_std_max
    ).to(device)

    # planner
    if cfg.planner.enabled:

        mppi = MPPI(
    planner_cfg=cfg.planner,
    action_dim=action_dim,
    device=device
)

        mpc = MPC(
    world_model=world_model,
    policy_prior=policy_prior,
    planner_cfg=cfg.planner,
    training_cfg=cfg.training,
    mppi=mppi
)

    else:
        mpc = None

    # trainer
    trainer = Trainer(
        env=env,
        encoder=encoder,
        simnorm=simnorm,
        world_model=world_model,
        policy_prior=policy_prior,
        mpc=mpc,
        cfg=cfg,
        device=device
    )

    # evaluation-only mode
    if cfg.evaluation.eval_only:
        evaluator = Evaluator(
            encoder=encoder,
            simnorm=simnorm,
            policy_prior=policy_prior,
            mpc=mpc,
            env=eval_env,
            cfg=cfg,
            device=device
        )
        evaluator.run()
        return
    
    # training loop
    print("Starting training...")

    while trainer.total_steps < cfg.training.steps:

        print(
        f"[MAIN] step={trainer.total_steps:,} / {cfg.training.steps:,}"
    )
        trainer.train(cfg.evaluation.eval_freq)

        print(
        f"[MAIN] finished train chunk → step={trainer.total_steps:,}"
    )

        evaluator = Evaluator(
            encoder=encoder,
            simnorm=simnorm,
            policy_prior=policy_prior,
            mpc=mpc,
            env=eval_env,
            cfg=cfg,
            device=device,
            csv_logger=csv_logger,
            eval_step=trainer.total_steps
        )

        evaluator.run()

    print("Training complete.")


if __name__ == "__main__":
    main()
