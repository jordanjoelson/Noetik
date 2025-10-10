import torch
import torch.nn.functional as F


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile regression loss.

    Args:
      diff: target − prediction (for us: q_min − V). If diff>0, V is lower than target.
      tau: expectile in (0,1). 0.5 is symmetric; >0.5 weights positive diffs more.

    Formula:  weight * diff^2,  where  weight = tau if diff>=0 else (1-tau)
    Intuition: with tau=0.7, underestimates (diff>0) are penalized 0.7x, overestimates 0.3x.
    """
    weight = torch.where(diff < 0, 1 - tau, tau)
    return weight * (diff ** 2)


def compute_iql_losses(
    *,
    obs,
    act,
    rew,
    next_obs,
    done,
    discount: float,
    tau: float,
    temperature: float,
    policy,
    value,
    q1,
    q2,
):
    """Compute scalar losses for IQL and return them plus logging stats.

    STEP-BY-STEP (this is the core learning logic):
    1) Build a TD target for Q:
       target_q = r + gamma*(1-done)*V(next_s)  (no grad)
    2) Make both Q networks fit this target via MSE.
    3) Fit V(s) to an *expectile* of Q(s,a) using min(Q1, Q2) for stability.
    4) Update the policy to put more probability mass on high-advantage actions using
       exponential weights w = exp(A/temperature), where A = min(Q1,Q2) − V(s).
    """
    # 1) TD target for Q (no gradients into the target)
    with torch.no_grad():
        v_next = value(next_obs)                      # V(next_s)
        target_q = rew + discount * (1 - done) * v_next

    # 2) Q losses: both Q1 and Q2 fit the same target
    q1_pred = q1(obs, act)
    q2_pred = q2(obs, act)
    loss_q1 = F.mse_loss(q1_pred, target_q)
    loss_q2 = F.mse_loss(q2_pred, target_q)
    loss_q = loss_q1 + loss_q2

    # 3) Value loss: expectile regression against q_min (no gradients from q into V's target)
    with torch.no_grad():
        q_min = torch.minimum(q1_pred, q2_pred)
    v_pred = value(obs)
    diff = q_min - v_pred
    loss_v = expectile_loss(diff, tau).mean()

    # 4) Policy loss: advantage-weighted log-likelihood
    #    A = q_min - v_pred (no-grad); weights = exp(A/temperature)
    with torch.no_grad():
        advantage = (q_min - v_pred).clamp(max=100.0)  # keep numbers sane
        weights = torch.exp(advantage / max(temperature, 1e-8)).clamp(max=100.0)

    # Sample actions and get their log-prob under current policy
    _, logp, _, _ = policy(obs, deterministic=False)
    loss_policy = (-(weights * logp)).mean()

    # Helpful scalars to print or log
    logs = {
        'q1': q1_pred.mean().item(),
        'q2': q2_pred.mean().item(),
        'v': v_pred.mean().item(),
        'adv_mean': advantage.mean().item(),
        'weight_mean': weights.mean().item(),
        'target_q_mean': target_q.mean().item(),
        'loss_q': loss_q.item(),
        'loss_v': loss_v.item(),
        'loss_policy': loss_policy.item(),
    }

    return loss_policy, loss_q, loss_v, logs

