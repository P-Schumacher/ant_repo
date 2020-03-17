import numpy as np
import gym
import sys
import time
import math
import tensorflow as tf
import wandb

from agent_files.HIRO import HierarchicalAgent
from utils.logger import Logger
from utils.utils import create_world
def maybe_verbose_output(t, agent, env, action, cnf, state):
    if cnf.render:
        print(f"action: {action}")
        print(f"time is: {t}")
        if not cnf.flat_agent:
            print(f"goal: {agent.goal}")
            if agent.meta_time and cnf.render:
                print(f"GOAL POSITION: {agent.goal}")
                if agent.goal_type == 'Direction':
                    env.set_goal(state[:3] + agent.goal[:3])
                else:
                    env.set_goal(agent.goal[:3])

def main(cnf):
    env, agent = create_world(cnf)
    cnf = cnf.main
    # create objects 
    logger = Logger(cnf.log, cnf.time_limit)
    # Load previously trained model.
    if cnf.load_model: agent.load_model(f'./experiments/models/{agent._file_name}')
    # Training loop
    state, done = env.reset(), False
    for t in range(int(cnf.max_timesteps)):
        action = agent.select_noisy_action(state)
        maybe_verbose_output(t, agent, env, action, cnf, state)
        next_state, reward, done, _ = env.step(action)
        intr_rew = agent.replay_add(state, action, reward, next_state, done)
        if t > cnf.start_timesteps and not t % cnf.train_every:
            agent.train(t)
        state = next_state
        logger.inc(t, reward)

        if done:
            print(f"Total T: {t+1} Episode Num: {logger.episode_num+1}+ Episode T: {logger.episode_timesteps} Reward: {logger.episode_reward}")
            # Reset environment
            agent.reset()
            hard_reset = logger.log(t, intr_rew)
            logger.reset()
            state, done = env.reset(), False
        # Evaluate episode
        if (t + 1) % cnf.eval_freq == 0:
            avg_ep_rew, avg_intr_rew, success_rate = agent.evaluation(env)
            state, done = env.reset(), False
            agent.reset()
            logger.reset(post_eval=True)
            logger.log_eval(t, avg_ep_rew, avg_intr_rew, success_rate)
            if cnf.save_model: agent.save_model(f'./experiments/models/{agent._file_name}')

