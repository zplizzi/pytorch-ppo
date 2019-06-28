from collections import deque
import multiprocessing as mp

import gym
import numpy as np
import torch
from torchvision import transforms

from running_mean_std import apply_normalizer

class GamePlayer:
    """A manager class for running multiple game-playing processes."""
    def __init__(self, args, shared_obs):
        self.episode_length = deque(maxlen=100)
        self.episode_rewards = deque(maxlen=100)

        # Start game-playing processes
        self.processes = []
        for i in range(args.num_workers):
            parent_conn, child_conn = mp.Pipe()
            worker = SubprocWorker(i, child_conn, args, shared_obs)
            ps = mp.Process(target=worker.run)
            ps.start()
            self.processes.append((ps, parent_conn))

    def run_rollout(self, args, shared_obs, rewards, discounted_rewards, values,
                    policy_probs, actions, model, obs_normalizer, device,
                    episode_ends):
        model.eval()
        # Start with the actions selected at the end of the previous iteration
        step_actions = actions[:, -1]
        # Same with obs
        shared_obs[:, 0] = shared_obs[:, -1]
        for step in range(args.num_steps):
            # Apply normalization to all steps except the first (which was
            # already normalized)
            obs = shared_obs[:, step]
            if step != 0:
                if len(obs.shape) == 2:
                    obs = apply_normalizer(obs, obs_normalizer)
                    shared_obs[:, step] = obs

            # run the model
            obs_torch = torch.tensor(obs).to(device).float()
            step_values, dist = model(obs_torch)

            # Sample actions from the policy distribution
            step_actions = dist.sample()
            step_policy_probs = dist.log_prob(step_actions)

            # Store data for use in training
            step_actions = step_actions.detach().cpu().numpy()
            values[:, step] = step_values.detach().cpu().numpy().flatten()
            policy_probs[:, step] = step_policy_probs.detach().cpu().numpy()
            actions[:, step] = step_actions

            # Send the selected actions to workers and request a step
            for j, (p, pipe) in enumerate(self.processes):
                pipe.send(("step", step, step_actions[j]))

            # Receive step data from workers
            for j, (p, pipe) in enumerate(self.processes):
                (reward, discounted_reward, done, info) = pipe.recv()
                rewards[j, step] = reward
                discounted_rewards[j, step] = discounted_reward
                episode_ends[j, step] = done
                try:
                    self.episode_length.append(info['final_episode_length'])
                    self.episode_rewards.append(info['final_episode_rewards'])
                except KeyError:
                    pass


class SubprocWorker:
    """A worker for running an environment, intended to be run on a separate
    process."""
    def __init__(self, index, pipe, args, shared_obs):
        self.index = index
        self.pipe = pipe
        self.episode_steps = 0
        self.episode_rewards = 0
        self.disc_ep_rewards = 0
        self.previous_lives = 0
        self.args = args
        self.shared_obs = shared_obs

        self.env = gym.make(args.env_name)
        self.env.reset()

        # Data preprocessing for raw Atari frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            # Converts to tensor and from [0,255] to [0,1]
            transforms.ToTensor(),
            # For a tensor in range (0, 1), this will convert to range (-1, 1)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def run(self):
        """The worker entrypoint, will wait for commands from the main
        process and execute them."""
        try:
            while True:
                cmd, t, action = self.pipe.recv()
                if cmd == 'step':
                    self.pipe.send(self.step(action, t))
                elif cmd == 'close':
                    self.pipe.send(None)
                    break
                else:
                    raise RuntimeError('Got unrecognized cmd %s' % cmd)
        except KeyboardInterrupt:
            print('worker: got KeyboardInterrupt')
        finally:
            self.env.close()

    def step(self, action, t):
        """Perform a single step of the environment."""
        info = {}
        step_reward = 0
        # We have the option to skip steps_to_skip steps, selecting the same
        # action this many times and returning the cumulative rewards
        # from the skipped steps and the final observation
        for _ in range(self.args.steps_to_skip):
            obs, reward, done, _ = self.env.step(action)
            fake_done = done

            # DQN used this "cheat" where we check if the agent lost
            # a life, and if so we indicate the episode to be over
            # (but don't actually reset the environment unless it
            # really is over)
            if self.args.end_on_life_loss:
                lives = self.env.ale.lives()
                if (self.previous_lives > lives and lives > 0):
                    # We died
                    fake_done = True
                self.previous_lives = lives

            self.episode_rewards += reward
            step_reward += reward

            if done:
                info["final_episode_length"] = self.episode_steps
                info["final_episode_rewards"] = self.episode_rewards
                obs = self.env.reset()

                # This breaks the Box2d games but should try adding it back for
                # Atari.
                # # perform a number of random steps after reset
                # for _ in range(np.random.randint(0, 30)):
                #     obs, _, _done, _ = self.env.step(action)
                #     if _done:
                #         obs = self.env.reset()

                self.episode_steps = 0
                self.episode_rewards = 0

            self.episode_steps += 1

            # We store the observation in t+1 because it's really the next
            # step's observation
            if t < self.args.num_steps - 1:
                if self.args.model == "cnn":
                    obs = self.transform(obs).numpy()
                self.shared_obs[self.index, t+1] = obs

            if done or fake_done:
                # Stop skipping steps and just finish this step
                break

        if self.args.clip_rewards:
            # clip reward to one of {-1, 0, 1}
            step_reward = np.sign(step_reward)

        self.disc_ep_rewards = self.disc_ep_rewards * self.args.gamma \
            + step_reward
        last_disc_reward = self.disc_ep_rewards
        if done or fake_done:
            self.disc_ep_rewards = 0

        return step_reward, last_disc_reward, fake_done, info
