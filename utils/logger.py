import wandb

class Logger:
    def __init__(self, log, time_limit):
        '''Helper class that logs variables using wandb.
        It counts the episode_reward, the timesteps it took and the 
        number of episodes.
        :param log: Should we log things at all.
        :return: None'''
        self.logging = log
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_num = 0
        

    def inc(self, t, reward):
        '''Logs variables that should be logged every timestep, not 
        just at the env of every episode.
        :param reward: The one timestep reward to be logged.
        :param t: The current timestep. We do not keep track of it inside
        the logger to ensure consistency across files.
        :return: None'''
        self.episode_timesteps += 1
        self.episode_reward += reward
        if self.logging:
            wandb.log({'step_rew': reward}, step = t)

    def reset(self, post_eval=False):
        '''Resets the logging values.
        :param post_eval: If True, increments the episode_num
        :return: None'''
        self.episode_timesteps = 0
        self.episode_reward = 0
        if not post_eval:
            self.episode_num += 1
    
    def log(self, t, intr_rew):
        '''Call this function at the end of an episode. The function arguments are logged by argument passage, the
        episode_reward is tracked internally.
        :param intr_rew: The episode intrinsic reward of the sub-agent.
        :param c_step: The current number of timesteps between subgoals.
        :return: None'''
        if self.logging:
            wandb.log({'ep_rew': self.episode_reward, 'intr_reward': intr_rew}, step = t)

    def log_eval(self, t, eval_rew, eval_intr_rew, eval_success):
        '''Log the evaluation metrics.
        :param t: The current timestep.
        :param eval_rew: The average episode reward of the evaluative episods.
        :param eval_intr_rew: The average intrinsic episode reward of the evaluative episodes.
        :param eval_success: The average success rate of the evaluative episodes.
        :return: None'''
        if self.logging:
            wandb.log({'eval/eval_ep_rew': eval_rew, 'eval/eval_intr_rew': eval_intr_rew,
                  'eval/success_rate': eval_success}, step = t)

