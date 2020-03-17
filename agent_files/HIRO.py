import tensorflow as tf
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt
from utils.replay_buffers import ReplayBuffer
from agent_files.Agent import Agent
import wandb

sub_Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'goal'])
meta_Transition = namedtuple('Transition', ['state', 'goal', 'done'])

class TransitBuffer(ReplayBuffer):
    '''This class can be used like a normal ReplayBuffer of a non Hierarchical agent. It stores generated goals internally and autonomously creates
    the correct sub- and meta-agent transitions. Just don't sample directly from it.'''
    def __init__(self, agent, sub_env_spec, meta_env_spec, subgoal_dim, target_dim, main_cnf, agent_cnf, buffer_cnf):
        self._prepare_parameters(main_cnf, agent_cnf, target_dim, subgoal_dim)
        self._prepare_buffers(buffer_cnf, sub_env_spec['state_dim'], meta_env_spec['state_dim'],
                              sub_env_spec['action_dim'])
        self._prepare_control_variables()
        self._prepare_offpolicy_datastructures(sub_env_spec)
        self._agent = agent
            
    def add(self, state, action, reward, next_state, done):
        '''Adds the transitions to the appropriate buffers. Goal is saved at
        each timestep to be used for the transition of the next timestep. At 
        an episode end, the meta agent recieves a transition of t:t_end, 
        regardless if it has been c steps.'''
        self._timestep += 1
        if self._needs_reset:
            raise Exception("You need to reset the agent if a 'done' has occurred in the environment.")
        if not self._init:  # Variables are saved in the first call, transitions constructed in later calls.
            self._initialize_buffer(state, action, reward, next_state, done) 
        else: 
            self._finish_sub_transition(self.goal, reward)
            self._save_sub_transition(state, action, reward, next_state, done, self.goal)
            if self.meta_time:
                self._finish_meta_transition(state, done)
                self._save_meta_transition(state, self.goal, done)
                self._sum_of_rewards = 0
            if done:
                self._ep_rewards = 0
                self._agent.select_action(next_state) # This computes the next goal in the transitbuffer
                self._finish_sub_transition(self.goal, reward)
                self._finish_meta_transition(next_state, done)
                self._needs_reset = True
                return self._ep_rewards
            self._sum_of_rewards += reward

    def _initialize_buffer(self, state, action, reward, next_state, done):
        if done:
            self._finish_one_step_episode(state, action, reward, next_state, done)
            self._needs_reset = True
            return None
        self._save_sub_transition(state, action, reward, next_state, done, self.goal)
        self._save_meta_transition(state, self.goal, done)
        self._sum_of_rewards += reward
        self._init = True
        return None

    def compute_intr_reward(self, goal, state, next_state):
        '''Computes the intrinsic reward for the sub agent. It is the L2-norm between the goal and the next_state, restricted to those dimensions that
        the goal covers. In the HIRO Ant case: BodyPosition: x, y, z BodyOrientation: a, b, c, d JointPositions: 2 per leg. Total of 15 dims. State space
        also contains derivatives of those quantities (i.e. velocity). Total of 32 dims.'''
        dim = self._subgoal_dim
        if self.goal_type == "Absolute":
            rew = -tf.norm(next_state[:dim] - goal)  # complete absolute goal reward
        elif self.goal_type == "Direction":
            rew =  -tf.norm(state[:dim] + goal - next_state[:dim])  # complete directional reward
        else:
            raise Exception("Goal type has to be Absolute or Direction")
        return rew
    
    def _collect_seq_state_actions(self, state, action):
        '''These are collected so that the off policy correction for the meta-agent can
        be calculated in a more efficient way.'''
        if self._offpolicy:
            self._state_seq[self._ptr] = state 
            self._action_seq[self._ptr] = action 
            self._ptr += 1 
            
    def _finish_sub_transition(self, next_goal, reward): 
        old = self._load_sub_transition() 
        intr_reward = self.compute_intr_reward(old.goal, old.state, old.next_state) * self._sub_rew_scale 
        if self._ri_re: 
            intr_reward += reward
        self._ep_rewards += intr_reward
        self._add_to_sub(old.state, old.goal, old.action, intr_reward, old.next_state, next_goal, old.done)

    def _finish_meta_transition(self, next_state, done):
        old = self._load_meta_transition()
        self._add_to_meta(old.state, old.goal, self._sum_of_rewards * self._meta_rew_scale, next_state, done)

    def _finish_one_step_episode(self, state, action, reward, next_state, done):
        '''1 step episodes are handled separately because the adding of states to 
        the replay buffers is always one step behind the environment timestep.'''
        goal = self.goal
        self._agent.select_action(next_state)
        next_goal = self.goal
        intr_reward = self.compute_intr_reward(goal, state, next_state)
        self._add_to_sub(state, goal, action, intr_reward, next_state, next_goal, done)
        self._add_to_meta(state, goal, reward * self._meta_rew_scale, next_state, done)

    def _add_to_sub(self, state, goal, action, intr_reward, next_state, next_goal, extr_done):
        '''Adds the relevant transition to the sub-agent replay buffer.'''
        if self._offpolicy:
            self._collect_seq_state_actions(state, action)
        # Remove target from sub-agent state. Conform with paper code.
        state = state[:-self._target_dim]  
        next_state = next_state[:-self._target_dim]
        cat_state = np.concatenate([state, goal])
        cat_next_state = np.concatenate([next_state, next_goal])
        # Zero out x and y for sub-agent. Conform with paper code.
        # Makes only sense with ee_pos or torso_pos, not joints
        if self._zero_obs:
            cat_state[:self._zero_obs] = 0
            cat_next_state[:self._zero_obs] = 0
        self._sub_replay_buffer.add(cat_state, action, intr_reward,
                                   cat_next_state, extr_done, 0, 0)
        
    def _save_sub_transition(self, state, action, reward, next_state, done, goal):
        self.sub_transition = sub_Transition(state, action, reward, next_state, done, goal)

    def _load_sub_transition(self):
        return self.sub_transition

    def _add_to_meta(self, state, goal, sum_of_rewards, next_state_c, done_c):
        self._meta_replay_buffer.add(state, goal, sum_of_rewards, next_state_c, done_c, self._state_seq,
                                    self._action_seq)
        self._reset_sequence()

    def _reset_sequence(self):
        '''After the sequence has been added to the meta replaybuffer, we overwrite the arrays with np.inf,
        those are handled in the offpolicy correction correctly. This enables us to have variable length 
        state and action sequences.'''
        if self._offpolicy:
            self._state_seq[:] = np.inf
            self._action_seq[:] = np.inf
            self._ptr = 0
    
    def _save_meta_transition(self, state, goal, done):
        self.meta_transition = meta_Transition(state, goal, done)

    def _load_meta_transition(self):
        return self.meta_transition

    def _prepare_offpolicy_datastructures(self, sub_env_spec):
        # We dont want to create these arrays for large c_step if we do not correct
        c_step = [1 if not self._offpolicy else self._c_step][0]
        self._state_seq = np.ones(shape=[c_step, sub_env_spec['state_dim'] - self._subgoal_dim + self._target_dim]) * np.inf 
        self._action_seq = np.ones(shape=[c_step, sub_env_spec['action_dim']]) * np.inf 

    def _prepare_control_variables(self):
        self._sum_of_rewards = 0
        self._ep_rewards = 0
        self._init = False  # The buffer.add() does not create a transition in s0 as we need s1
        self.meta_time = False
        self._needs_reset = False
        self._timestep = -1
        self._ptr = 0

    def _prepare_buffers(self, buffer_cnf, sub_state_dim, meta_state_dim, action_dim):
        assert sub_state_dim == meta_state_dim - self._target_dim + self._subgoal_dim
        self._sub_replay_buffer = ReplayBuffer(sub_state_dim, action_dim, **buffer_cnf)
        self._meta_replay_buffer = ReplayBuffer(meta_state_dim, self._subgoal_dim, **buffer_cnf)

    def _prepare_parameters(self, main_cnf, agent_cnf, target_dim, subgoal_dim):
        # Env specs
        self._offpolicy = main_cnf.offpolicy
        self._target_dim = target_dim
        self._subgoal_dim = subgoal_dim
        # Main cnf
        self._c_step = main_cnf.c_step
        # Agent cnf
        self._zero_obs = agent_cnf.zero_obs
        self.goal_type = agent_cnf.goal_type
        self._sub_rew_scale = agent_cnf.sub_rew_scale
        self._meta_rew_scale = agent_cnf.meta_rew_scale
        self._ri_re = agent_cnf.ri_re


class HierarchicalAgent(Agent):
    def __init__(self, agent_cnf, buffer_cnf, main_cnf, env_spec, model_cls, subgoal_dim):
        self._prepare_parameters(agent_cnf, main_cnf, env_spec, subgoal_dim)
        self._file_name = self._create_file_name(main_cnf.model, main_cnf.env, main_cnf.descriptor)
        
        meta_env_spec, sub_env_spec = self._build_modelspecs(env_spec)
        self._transitbuffer = TransitBuffer(self, sub_env_spec, meta_env_spec, subgoal_dim, self._target_dim, main_cnf, agent_cnf,
                                           buffer_cnf)
        self._sub_agent = model_cls(**sub_env_spec, **agent_cnf.sub_model)
        self._meta_agent = model_cls(**meta_env_spec, **agent_cnf.meta_model)
        # Logic variables
        # Set this to its maximum, such that we query the meta agent in the first iteration
        self._goal_counter = self._c_step 
        self._evals = 0  
        
    def select_action(self, state):
        '''Selects an action from the sub agent to output. For this a goal is queried from the meta agent and
        saved(!) for the add-fct. In this function, no noise is added.'''
        self._get_meta_goal(state)
        action = self._get_sub_action(state)
        return action

    def select_noisy_action(self, state):
        '''Selects an action from sub- and meta-agent and adds gaussian noise to it. Then clips actions appropriately to the sub.max_action
        or the META_RANGES'''
        action = self.select_action(state) + self._gaussian_noise(self._sub_noise, self._action_dim)
        if self.meta_time:
            self.goal = self.goal + self._gaussian_noise(self._meta_noise, self._subgoal_dim)
            if not self._spherical_coord:
                self.goal = tf.clip_by_value(self.goal, -self._subgoal_ranges, self._subgoal_ranges)
        return tf.clip_by_value(action, -self._sub_agent._max_action, self._sub_agent._max_action)
    
    def train(self, timestep):
        '''Train the agent with 1 minibatch. The meta-agent is trained every c_step steps.'''
        sub_avg = np.zeros([6,], dtype=np.float32)
        for i in range(self._gradient_steps):
            *metrics, = self._sub_agent.train(self.sub_replay_buffer, self._batch_size, timestep, self._log)
            sub_avg += metrics
        sub_avg /= self._gradient_steps

        meta_avg = np.zeros([6,], dtype=np.float32)
        for i in range(int(self._gradient_steps / 10)):
            *metrics, = self._meta_agent.train(self.meta_replay_buffer, self._batch_size, timestep, self._log,
                                              self._sub_agent.actor)
            meta_avg += metrics
        meta_avg /= int(self._gradient_steps / 10)
        if self._log:
            for avg, name in zip([sub_avg, meta_avg], ['sub', 'meta']):
                wandb.log({f'{name}/actor_loss': avg[0],
                           f'{name}/critic_loss': avg[1],
                           f'{name}/critic_gradnorm': avg[2],
                           f'{name}/actor_gradnorm': avg[3], 
                           f'{name}/actor_gradstd': avg[4],
                           f'{name}/critic_gradstd': avg[5]}, step = timestep)

    def replay_add(self, state, action, reward, next_state, done):
        return self._transitbuffer.add(state, action, reward, next_state, done)
        
    def save_model(self, string):
        '''Saves the weights of sub and meta agent to a file.'''
        self._sub_agent.actor.save_weights(string + "_sub_actor")
        self._sub_agent.critic.save_weights(string + "_sub_critic")
        self._meta_agent.actor.save_weights(string + "_meta_actor")
        self._meta_agent.critic.save_weights(string + "_meta_critic")

    def load_model(self, string):
        '''Loads the weights of sub and meta agent from a file.'''
        self._sub_agent.actor.load_weights(string + "_sub_actor")
        self._sub_agent.critic.load_weights(string + "_sub_critic")
        self._meta_agent.actor.load_weights(string + "_meta_actor")
        self._meta_agent.critic.load_weights(string + "_meta_critic")

    def reset(self):
        '''Want this reset such that the meta agent proposes the first goal every
        episode and we don't get the last goal from the previous episode. It also
        prevents lingering old transition components from being used. This also resets
        the sum of rewards for the meta agent.'''
        self._goal_counter = self._c_step
        self._transitbuffer._init = False
        self._transitbuffer._sum_of_rewards = 0
        self._transitbuffer._needs_reset = False
        self._transitbuffer._state_seq[:] = np.inf
        self._transitbuffer._action_seq[:] = np.inf
        self._transitbuffer._ptr = 0

    def _prepare_parameters(self, agent_cnf, main_cnf, env_spec, subgoal_dim):
        # Env specs
        self._subgoal_ranges = np.array(env_spec['subgoal_ranges'], dtype=np.float32)
        self._target_dim = env_spec['target_dim']
        self._action_dim = env_spec['action_dim']
        self._subgoal_dim = subgoal_dim
        # Main cnf
        self._batch_size = main_cnf.batch_size
        self._c_step = main_cnf.c_step
        self._seed = main_cnf.seed
        self._log = main_cnf.log
        self._gradient_steps = main_cnf.gradient_steps
        self._visit = main_cnf.visit
        # Agent cnf
        self._spherical_coord = agent_cnf.spherical_coord
        self._num_eval_episodes = agent_cnf.num_eval_episodes
        self._meta_mock = agent_cnf.meta_mock
        self._sub_mock = agent_cnf.sub_mock
        self._meta_noise = agent_cnf.meta_noise
        self._sub_noise = agent_cnf.sub_noise
        self._zero_obs = agent_cnf.zero_obs
        self.goal_type = agent_cnf.goal_type

    def _maybe_mock(self, goal):
        '''Replaces the subgoal by a constant goal that is put in by hand. For debugging and understanding.'''
        if not self._meta_mock:
            return goal
        mock_goal = np.random.uniform(0, 0.1, size=[3,]) 
        return mock_goal
    
    def _maybe_move_over_table(self, goal, move=False):
        '''Adds a constant term to the generated subgoal such that it is forced to be 
        above the table.'''
        if move:
            print("moved")
            goal = tf.constant([0,0,1,0,0,0], dtype=tf.float32) + goal 
        return goal 
    
    def _build_modelspecs(self, env_spec):
        '''Prepares the keyword arguments for the TD3 algo for the meta and the sub agent. The meta agent outputs
        an action corresponding to the META_RANGES dimension. The state space of the sub agent is the sum of its 
        original state space and the action space of the meta agent.'''
        meta_env_spec = env_spec.copy()
        sub_env_spec = env_spec.copy()
        meta_env_spec['action_dim'] = len(self._subgoal_ranges)
        meta_env_spec['max_action'] = self._subgoal_ranges
        sub_env_spec['state_dim'] = sub_env_spec['state_dim'] - self._target_dim + meta_env_spec['action_dim']
        return meta_env_spec, sub_env_spec

    def _get_meta_goal(self, state):
        self.goal, self.meta_time = self._sample_goal(state)
        self._maybe_spherical_coord_trafo()

    def _maybe_spherical_coord_trafo(self):
        '''If we give a meta-goal in spherical coordinates, this transforms it back to cartesian
        coordinates.
        r = self.goal[0]
        theta = self.goal[1]
        phi = self.goal[2]
        '''
        if self._spherical_coord:
            # Output of meta is goal \in [-1,1] Now bring it in [0,1] for spherical coordinates
            r = 1.867/2 - 0.1
            self.goal = (self.goal - (-1)) / (1 - (-1))
            x = self.goal[0] * r * np.sin(self.goal[1] * np.pi) * np.cos(self.goal[2] * np.pi)
            y = self.goal[0] * r * np.sin(self.goal[1] * np.pi) * np.sin(self.goal[2] * np.pi)
            z = self.goal[0] * r * np.cos(self.goal[1] * np.pi)
            x += 0.622
            y -= 0.605
            z += 0.86
            self.goal = tf.constant(np.array([x, y, z], dtype=np.float32))

    def _get_sub_action(self, state):
        if self._sub_mock:
            action = np.zeros([8,], dtype=np.float32)
            action[:7] = self.goal
            return action 
        # Zero out x,y for sub agent. Then take target away in select_action. Conform with HIRO paper.
        if self._zero_obs:
            state = state.copy()
            state[:self._zero_obs] = 0
        return  self._sub_agent.select_action(np.concatenate([state[:-self._target_dim], self.goal]))
    
    def _goal_transition_fn(self, goal, previous_state, state):
        '''When using directional goals, we have to transition the goal at every 
        timestep to keep it at the same absolute position for n timesteps. In absolute 
        mode, this is just the identity.'''
        if self.goal_type == "Absolute":
            return goal
        elif self.goal_type == "Direction":
            dim = self._subgoal_dim
            return previous_state[:dim] + goal - state[:dim]
        else:
            raise Exception("Enter a valid type for the goal, Absolute or Direction.")
    
    def _check_inner_done(self, state, next_state, goal):
        '''Checks if the sub-agent has managed to reach the subgoal and then calls a new subgoal.'''
        inner_done = self._inner_done_cond(state, next_state, goal)
        if self._log:
            wandb.log({'distance_to_goal':inner_done}, commit=False)
    
    def _inner_done_cond(self, state, next_state, goal):
        dim = self._subgoal_dim
        if self.goal_type == 'Absolute':
            diff = next_state[:dim] - goal[:dim]
        else:
            diff = state[:dim] + goal - next_state[:dim]
        return tf.norm(diff) 

    def _sample_goal(self, state):
        '''Either output the existing goal or query the meta agent for a new goal, depending on the timestep. The 
        goal and the meta_time boolean are saved to the transitbuffer for later use by the add-fct.'''
        self._goal_counter += 1
        if self._goal_counter < self._c_step:
            meta_time = False
            goal = self._goal_transition_fn(self.goal, self._prev_state, state)
            self._prev_state = state
        else:
            self._goal_counter = 0
            self._prev_state = state
            meta_time = True
            goal = self._meta_agent.select_action(state)
            goal = self._maybe_move_over_table(goal)
            goal = self._maybe_mock(goal)
        self._check_inner_done(self._prev_state, state, goal)
        return goal, meta_time
    
    def _eval_policy(self, env, seed, visit):
        '''Runs policy for X episodes and returns average reward, average intrinsic reward and success rate.
        Different seeds are used for the eval environments. visit is a boolean that decides if we record visitation
        plots.'''
        env.seed(self._seed + 100)
        avg_ep_reward = []
        avg_intr_reward = []
        success_rate = 0
        visitation = np.zeros((env.max_episode_steps, env.observation_space.shape[0]))
        for episode_nbr in range(self._num_eval_episodes):
            print(f'eval number: {episode_nbr} of {self._num_eval_episodes}')
            step = 0
            state, done = env.reset(evalmode=True), False
            self.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                avg_ep_reward.append(reward)
                avg_intr_reward.append(self._transitbuffer.compute_intr_reward(self.goal, state, next_state))
                state = next_state
                if visit:
                    visitation[step, :] = state
                step += 1
                if done and step < env.max_episode_steps:
                    success_rate += 1
            if visit:
                np.save('./results/visitation/visitation_{self._evals}_{episode_nbr}_{self._file_name}', visitation)

        avg_ep_reward = np.sum(avg_ep_reward) / self._num_eval_episodes
        avg_intr_reward = np.sum(avg_intr_reward) / self._num_eval_episodes
        success_rate = success_rate / self._num_eval_episodes
        print("---------------------------------------")
        print(f'Evaluation over {self._num_eval_episodes} episodes: {avg_ep_reward}')
        print("---------------------------------------")
        self._evals += 1
        return avg_ep_reward, avg_intr_reward, success_rate

    @property
    def goal(self):
        return self._transitbuffer.goal

    @goal.setter
    def goal(self, goal):
        self._transitbuffer.goal = goal

    @property
    def meta_time(self):
        return self._transitbuffer.meta_time

    @meta_time.setter
    def meta_time(self, meta_time):
        self._transitbuffer.meta_time = meta_time

    @property
    def sub_replay_buffer(self):
        return self._transitbuffer._sub_replay_buffer

    @property
    def meta_replay_buffer(self):
        return self._transitbuffer._meta_replay_buffer
