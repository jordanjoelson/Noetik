from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

from src.iql.config import IQLConfig
from src.iql.networks import GaussianTanhPolicy, ValueNet, QNet
from src.iql.losses import compute_iql_losses
from src.iql.batch import Batch
 


class IQLAgent(nn.Module):
    def __init__(self, cfg: IQLConfig, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Initialize networks and move to device
        self.policy = GaussianTanhPolicy(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes,
                                         cfg.policy_log_std_min, cfg.policy_log_std_max).to(device)
        self.value = ValueNet(cfg.obs_dim, cfg.hidden_sizes).to(device)
        self.q1 = QNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes).to(device)
        self.q2 = QNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes).to(device)

        self.opt_policy = optim.Adam(self.policy.parameters(), lr=cfg.lr_policy, weight_decay=cfg.weight_decay)
        self.opt_q = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr_q, weight_decay=cfg.weight_decay)
        self.opt_v = optim.Adam(self.value.parameters(), lr=cfg.lr_v, weight_decay=cfg.weight_decay)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action for deployment/evaluation.

        deterministic=True returns tanh(mu) (no sampling). Use deterministic=False
        if you want exploration-like behavior.
        """
        obs = obs.to(self.device)
        a, _, _, _ = self.policy(obs, deterministic=deterministic)
        return a

    def update(self, batch: Batch) -> Dict[str, float]:
        """Run one gradient step of IQL on a batch of data.

        Returns a dict of scalars you can print or log.
        """
        # Move batch to device
        batch.obs = batch.obs.to(self.device)
        batch.act = batch.act.to(self.device)
        batch.rew = batch.rew.to(self.device)
        batch.next_obs = batch.next_obs.to(self.device)
        batch.done = batch.done.to(self.device)
        
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
        # compute grad norm for q before clipping
        q_grad_norm = 0.0
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            if p.grad is not None:
                q_grad_norm += p.grad.data.norm(2).item() ** 2
        q_grad_norm = q_grad_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        q_lr = self.opt_q.param_groups[0]['lr'] if len(self.opt_q.param_groups) > 0 else None
        self.opt_q.step()

        # Step V
        self.opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        v_grad_norm = 0.0
        for p in self.value.parameters():
            if p.grad is not None:
                v_grad_norm += p.grad.data.norm(2).item() ** 2
        v_grad_norm = v_grad_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), 10.0)
        v_lr = self.opt_v.param_groups[0]['lr'] if len(self.opt_v.param_groups) > 0 else None
        self.opt_v.step()

        # Step policy
        self.opt_policy.zero_grad(set_to_none=True)
        loss_policy.backward()
        policy_grad_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                policy_grad_norm += p.grad.data.norm(2).item() ** 2
        policy_grad_norm = policy_grad_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        policy_lr = self.opt_policy.param_groups[0]['lr'] if len(self.opt_policy.param_groups) > 0 else None
        self.opt_policy.step()

        # Try to log per-update diagnostics to Weights & Biases if available
        try:
            import wandb
            wandb.log({
                'loss_q': loss_q.item(),
                'loss_v': loss_v.item(),
                'loss_policy': loss_policy.item(),
                'grad_norm_q': q_grad_norm,
                'grad_norm_v': v_grad_norm,
                'grad_norm_policy': policy_grad_norm,
                'lr_q': q_lr,
                'lr_v': v_lr,
                'lr_policy': policy_lr,
            })
        except Exception:
            # wandb not installed or logging not desired — ignore
            pass

        return logs