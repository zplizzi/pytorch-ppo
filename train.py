import numpy as np
import torch

def train_step(model, optim, batch_data, args, i, tracker):
    model.train()
    optim.zero_grad()

    advantages, rewards_to_go, values, actions, obs, \
        selected_prob = batch_data

    values_new, dist_new = model(obs)
    values_new = values_new.flatten()
    selected_prob_new = dist_new.log_prob(actions)

    # Compute the PPO loss
    prob_ratio = torch.exp(selected_prob_new) / torch.exp(selected_prob)

    a = prob_ratio * advantages
    b = torch.clamp(prob_ratio, 1 - args.epsilon, 1 + args.epsilon) * advantages
    ppo_loss = -1 * torch.mean(torch.min(a, b))

    # Compute the value function loss
    # Clipped loss - same idea as PPO loss, don't allow value to move too
    # far from where it was previously
    value_pred_clipped = values + (values_new - values).clamp(-args.epsilon, args.epsilon)
    value_losses = (values_new - rewards_to_go) ** 2
    value_losses_clipped = (value_pred_clipped - rewards_to_go) ** 2
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
    value_loss = value_loss.mean()

    entropy_loss = torch.mean(dist_new.entropy())

    loss = ppo_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
    optim.step()

    if i % 5 == 0:
        tracker.add_scalar("loss/ppo loss", ppo_loss, i)
        tracker.add_scalar("loss/value_loss", value_loss*args.value_loss_coef, i)
        tracker.add_scalar("loss/entropy_loss", -1*entropy_loss*args.entropy_coef, i)
        tracker.add_scalar("loss/loss", loss, i)
        tracker.add_scalar("training/mean value", torch.mean(values_new), i)
        tracker.add_scalar("training/mean rewards to go", torch.mean(rewards_to_go), i)
        tracker.add_scalar("training/mean prob ratio", torch.mean(prob_ratio), i)
        tracker.add_scalar("policy/policy entropy", entropy_loss, i)

    if i % 25 == 0:
        if len(actions.shape) == 2:
            for k in range(actions.shape[1]):
                tracker.add_histogram(f"policy/actions_{k}", actions[:, k], i)
        else:
            tracker.add_histogram("policy/actions", actions, i)

        tracker.add_histogram("training/values", values_new, i)
        tracker.add_histogram("training/advantages", advantages, i)
        tracker.add_histogram("training/rewards to go", rewards_to_go, i)
        tracker.add_histogram("training/prob ratio", prob_ratio, i)
        tracker.add_histogram("loss/ppo_loss_hist", torch.min(a, b), i)
        try:
            tracker.add_histogram("policy/var", model.dist.stds.exp().detach().cpu(), i)
        except AttributeError:
            pass
