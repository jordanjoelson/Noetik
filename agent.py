from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from src.iql.config import IQLConfig
from src.iql.networks import GaussianTanhPolicy, ValueNet, QNet
from src.iql.losses import compute_iql_losses
from src.iql.batch import Batch
 


class IQLAgent(nn.Module):
    def __init__(self, cfg: IQLConfig):
        super().__init__()
        self.cfg = cfg
        self.policy = GaussianTanhPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes,
                                         cfg.policy_log_std_min, cfg.policy_log_std_max)
        self.value = ValueNet(cfg.obs_dim, cfg.hidden_sizes)
        self.q1 = QNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes)
        self.q2 = QNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=cfg.lr_policy, weight_decay=cfg.weight_decay)
        self.opt_q = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q, weight_decay=cfg.weight_decay)
        self.opt_v = optim.Adam(self.value.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action for deployment/evaluation.

        deterministic=True returns tanh(mu) (no sampling). Use deterministic=False
        if you want exploration-like behavior.
        """
        a, _, _, _ = self.policy(obs, deterministic=deterministic)
        return a

    def update(self, batch: Batch) -> Dict[str, float]:
        """Run one gradient step of IQL on a batch of data.

        Returns a dict of scalars you can print or log.
        """
        cfg = self.cfg
        loss_policy, loss_q, loss_v, logs = compute_iql_losses(
            obs=batch.obs,
            act=batch.act,
            rew=batch.rew,
            next_obs=batch.next_obs,
            done=batch.done,
            discount=cfg.discount,
            tau=cfg.tau,
            temperature=cfg.temperature,
            policy=self.policy,
            value=self.value,
            q1=self.q1,
            q2=self.q2,
        )

        # Step Q
        self.opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.opt_q.step()

        # Step V
        self.opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
        self.opt_v.step()

        # Step policy
        self.opt_policy.zero_grad(set_to_none=True)
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.opt_policy.step()

        return logs