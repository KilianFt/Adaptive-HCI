import gymnasium as gym
import numpy as np

from users import MouseProportionalUser


class Environment(gym.Env):
    def __init__(self, max_steps):
        self.goal = None
        self.position = np.zeros(2)
        self.current_steps = None
        self.max_steps = max_steps
        self.reset()

    def step(self, action):
        self.position += action
        self.position = np.clip(self.position, np.array([-10, -10]), np.array([10, 10]))
        distance_from_goal = np.linalg.norm(self.position - self.goal)
        done = self.is_done()
        reward = -distance_from_goal
        reward = np.clip(reward, -1, 1)
        return self.get_state(), float(reward), done, {}

    def is_done(self):
        return np.linalg.norm(self.position - self.goal) < 1 or self.current_steps >= self.max_steps

    def get_state(self):
        return self.position

    def get_goal(self):
        return self.goal

    def reset(self, **kwargs):
        super(Environment, self).reset(**kwargs)
        self.goal = np.random.rand(2) * 10
        self.position = np.zeros(2)
        self.current_steps = 0
        return self.get_state()

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,))

    @property
    def action_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(2,))


class TwoDProjection(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "observation": self.observation_space["observation"],
            "desired_goal": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
            "achieved_goal": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64),
        })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    @staticmethod
    def _project_observation(observation):
        observation["desired_goal"] = observation["desired_goal"][:2]
        observation["achieved_goal"] = observation["achieved_goal"][:2]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._project_observation(observation)
        return observation, info

    def step(self, action):
        mujoco_aciton = np.zeros(4)
        mujoco_aciton[:2] = action
        old_obs = self.env.unwrapped._get_obs()
        # 0.05 is the scaling factor of the environment
        optimal_z = (old_obs["desired_goal"][2] - old_obs["achieved_goal"][2]) / 0.05
        mujoco_aciton[2] = optimal_z

        observation, reward, terminated, truncated, info = self.env.step(mujoco_aciton)
        self._project_observation(observation)
        if float(reward) > -0.005:
            terminated = True

        return observation, reward, terminated, truncated, info


class EnvironmentWithUser(gym.Wrapper):
    """
    This wrapper adds a user to the environment.

    An action goes into the environment, the user observes the environment and returns some features, along with the
    classical transition.
    """

    def __init__(self, env: gym.Env, user: MouseProportionalUser):
        super().__init__(env)
        self.user = user
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float64)

    def reset(self, **kwargs):
        env_obs, env_info = self.env.reset(**kwargs)
        user_observation, user_info = self.user.reset(env_obs, env_info)
        return user_observation, user_info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.user.step(observation, reward, terminated, truncated, info)

    def render(self):
        self.user.think()
        return self.env.render()
