import numpy as np
import gym

def get_gym_env_info(env_name):
    env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    num_obs = int(np.product(obs_shape))
    try:
        # discrete space
        num_actions = env.action_space.n
        action_type = "discrete"
    except AttributeError:
        # continuous space
        num_actions = env.action_space.shape[0]
        action_type = "continuous"
    return num_actions, obs_shape, num_obs, action_type
