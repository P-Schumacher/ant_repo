import os 
import sys
import numpy as np
import gym
import tensorflow as tf
from utils.replay_buffers import ReplayBuffer
import wandb 

class Agent:
    def __init__(self, agent_cnf, buffer_cnf, main_cnf, specs, model):
        self._prepare_params(agent_cnf, main_cnf)
        self._replay_buffer = ReplayBuffer(specs['state_dim'], specs['action_dim'], **buffer_cnf)
        self._file_name = self._create_file_name(main_cnf.model, main_cnf.env, main_cnf.descriptor)
        self._policy = model(**specs, **agent_cnf.sub_model) 
        
    def evaluation(self, env):
        '''Play N evaluation episodes where noise is turned off. We also evaluate only the [0,16] target, not a uniformly
        sampled one. The function then returns the avg reward, intrinsic reward and the success rate over the N episodes.'''
        # Set seed to clear tensorflow cache which prevents OutOfMemory error... I hate tensorflow
        tf.random.set_seed(self._seed)
        env.reset()
        avg_reward, avg_intr_reward, success_rate =  self._eval_policy(env, self._seed,
                                                                       self._visit)
        self.reset()
        return avg_reward, avg_intr_reward, success_rate

    def select_action(self, state):
        state = np.array(state)
        return self._policy.select_action(state) 

    def select_noisy_action(self, state):
        state = np.array(state)
        action = self._policy.select_action(state) + self._gaussian_noise(self._sub_noise)
        return tf.clip_by_value(action, -self._policy.max_action, self._policy.max_action)      
    
    def train(self, timestep):
        m_avg = np.zeros([6, ], dtype=np.float32)
        for i in range(self._gradient_steps):
            *metrics, = self._policy.train(self._replay_buffer, self._batch_size, timestep)
            m_avg += metrics 
        m_avg /= self._gradient_steps
        if self._log:
            wandb.log({f'sub/actor_loss': m_avg[0],
                       f'sub/critic_loss': m_avg[1],
                       f'sub/critic_gradnorm': m_avg[2],
                       f'sub/actor_gradnorm': m_avg[3], 
                       f'sub/actor_gradstd': m_avg[4],
                       f'sub/critic_gradstd': m_avg[5]}, step = timestep)

    def replay_add(self, state, action, next_state, reward, done):
        self._replay_buffer.add(state, action, next_state, self._sub_rew_scale * reward, done, 0, 0)

    def save_model(self, string):
        self._policy.actor.save_weights(string + "_policy_actor")
        self._policy.critic.save_weights(string + "_policy_critic")

    def load_model(self, string):
        self._policy.actor.load_weights(string + "_policy_actor")
        self._policy.critic.load_weights(string + "_policy_critic")

    def reset(self):
        pass # Not necessary for simple agent

    def _prepare_params(self, agent_cnf, main_cnf):
        self._num_eval_episodes = agent_cnf.num_eval_episodes
        self._sub_noise = agent_cnf.sub_noise
        self._sub_rew_scale = agent_cnf.sub_rew_scale
        self._seed = main_cnf.seed
        self._visit = main_cnf.visit
        self._log = main_cnf.log
        self._gradient_steps = main_cnf.gradient_steps
        self._batch_size = main_cnf.batch_size

    def _gaussian_noise(self, expl_noise, dimension=1):
        return np.random.normal(0, expl_noise, dimension) 

    def _create_file_name(self, policy, env, descriptor):
        '''Create file_name from experiment information to save model weights.'''
        file_name = f'{policy}_{env}_{descriptor}'
        print("---------------------------------------")
        print(f"Policy: {policy}, Env: {env}")
        print("---------------------------------------")
        return file_name

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def _eval_policy(self, env, seed, visit):
        '''Runs policy for X episodes and returns average reward, average intrinsic reward and success rate.
        Different seeds are used for the eval environments. visit is a boolean that decides if we record visitation
        plots.'''
        env.seed(seed + 100)
        avg_ep_reward = []
        success_rate = 0
        for episode_nbr in range(self._num_eval_episodes):
            print(f"eval number: {episode_nbr} of {self._num_eval_episodes}")
            step = 0
            state, done = env.reset(evalmode=True), False
            self.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                avg_ep_reward.append(reward)
                state = next_state
                step += 1
                if done and step < env.max_episode_steps:
                    success_rate += 1

        avg_ep_reward = np.sum(avg_ep_reward) / self._num_eval_episodes
        success_rate = success_rate / self._num_eval_episodes
        print("---------------------------------------")
        print(f'Evaluation over {self._num_eval_episodes} episodes: {avg_ep_reward}')
        print("---------------------------------------")
        return avg_ep_reward, 0, success_rate
