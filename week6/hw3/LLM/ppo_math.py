"""Auditable token-level clipped PPO and generalized advantage estimation."""
import torch


def gae(rewards, values, gamma=1.0, lam=.95):
    if rewards.shape!=values.shape or rewards.ndim!=1:raise ValueError('Expected equal one-dimensional trajectories')
    advantage=torch.zeros_like(rewards)
    carry=torch.zeros((),device=rewards.device)
    for t in reversed(range(len(rewards))):
        next_value=values[t+1] if t+1<len(values) else torch.zeros_like(carry)
        delta=rewards[t]+gamma*next_value-values[t]
        carry=delta+gamma*lam*carry
        advantage[t]=carry
    return advantage,advantage+values


def clipped_loss(logprobs, values, old_logprobs, old_values, advantages, returns,
                 clip=.2, value_clip=.2, value_coef=.5):
    log_ratio=logprobs-old_logprobs
    ratio=log_ratio.exp()
    policy=torch.maximum(-advantages*ratio,-advantages*ratio.clamp(1-clip,1+clip)).mean()
    clipped_values=old_values+(values-old_values).clamp(-value_clip,value_clip)
    value=.5*torch.maximum((values-returns).square(),(clipped_values-returns).square()).mean()
    total=policy+value_coef*value
    return total,{'policy_loss':policy.detach().item(),'value_loss':value.detach().item(),
                  'approx_kl':((ratio-1)-log_ratio).mean().detach().item(),
                  'clip_fraction':((ratio-1).abs()>clip).float().mean().detach().item()}
