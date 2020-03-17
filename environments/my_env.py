import numpy as np
import random
from gym import spaces

def get_goal_sample_fn(env_name, evalmode):
  if env_name == 'AntMaze':
    # NOTE: When evaluating (i.e. the metrics shown in the paper,
    # we use the commented out goal sampling function.  The uncommented
    # one is only used for training.
    if evalmode:
      return lambda: np.array([0., 16.]) # evaluation goal
    return lambda: np.random.uniform((-4, -4), (20, 20))
  elif env_name == 'AntPush':
    return lambda: np.array([0., 19.])
  elif env_name == 'AntFall':
    return lambda: np.array([0., 27., 4.5])
  else:
    assert False, 'Unknown env'


def get_reward_fn(env_name):
  if env_name == 'AntMaze':
    return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
  elif env_name == 'AntPush':
    return lambda obs, goal: -np.sum(np.square(obs[:2] - goal)) ** 0.5
  elif env_name == 'AntFall':
    return lambda obs, goal: -np.sum(np.square(obs[:3] - goal)) ** 0.5
  else:
    assert False, 'Unknown env'


def success_fn(last_reward):
  return last_reward > -5.0


class EnvWithGoal(object):

  def __init__(self, base_env, env_name, time_limit=None, render=False, simple_obs=False, evalmode=False):
    self.base_env = base_env
    self.goal_sample_fn = get_goal_sample_fn(env_name, evalmode)
    self.reward_fn = get_reward_fn(env_name)
    self.goal = None
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.base_env.observation_space.shape[0] + self.goal_sample_fn().shape[0],))
    self.step_count = 0
    self.max_episode_steps = time_limit
    self.render = render
    self.env_name = env_name
    self.target_dim = 2
    self.subgoal_dim = 15
    # cf. Hiro Paper
    self.subgoal_ranges = np.array([10, 10, 0.5, 1, 1, 1, 1, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3, 0.5, 0.3], dtype=np.float32)

  def reset(self, evalmode=False):
    self.step_count = 0
    obs = self.base_env.reset()
    if not evalmode:
        self.goal = self.goal_sample_fn()
    else:
        self.goal = get_goal_sample_fn(env_name=self.env_name, evalmode=True)()
    if self.render:
      self.set_target(self.goal)
    return np.concatenate([obs, self.goal])

  def step(self, a):
    self.step_count += 1 
    obs, _, done, info = self.base_env.step(a)
    if self.render:
      self.base_env.render()
    reward = self.reward_fn(obs, self.goal)
    done, _ = self._done_condition(done, reward)
    return np.concatenate([obs, self.goal]), reward, done, info

  def seed(self, seed):
    self.base_env.seed(seed)
    
  @property
  def action_space(self):
    return self.base_env.action_space

  def _done_condition(self, done, reward):
    cd1_end_of_episode = self.step_count == self.max_episode_steps
    cd2_success = success_fn(reward)
    if cd2_success:
      print("Success, Goal was: ({},{})".format(self.goal[0],self.goal[1]))
    if cd1_end_of_episode or cd2_success:
      done = True
    reward = reward if not done else 0
    return done, reward

  def observation_wrapper(self, obs):
    if not self.step_count % 10:
      self.posix = obs[0]
      self.posiy = obs[1]
    obs[0] = obs[0] - self.posix
    obs[1] = obs[1] - self.posiy

  def set_target(self, goal):
    print("Set target")
    self.base_env.wrapped_env.set_target(goal)

  def set_goal(self, goal):
    print("Set target")
    self.base_env.wrapped_env.set_goal(goal)

if __name__ == '__main__':
  env_name = "AntMaze"
  from environments.create_maze_env import create_maze_env
  env = create_maze_env(env_name, render=False)
  env = EnvWithGoal(env, env_name, 500, render=False, evalmode=False)
  done = env.reset()
  for i in range(1000):
    obs, reward, done, _ = env.step(env.action_space.sample())
    print(done)
