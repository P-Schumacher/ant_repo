import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
    '''Simple replay buffer class which samples tensorflow tensors.'''
    # TODO max_size and c_step from command prmpt
    def __init__(self, state_dim, action_dim, c_step, offpolicy, max_size):
        if not offpolicy:
            c_step = 1
        self.max_size = max_size
        self.ptr = 0
        self.size = 0 
        self.action_dim = action_dim
        self.state = np.zeros((max_size, state_dim), dtype = np.float32)
        self.action = np.zeros((max_size, action_dim), dtype = np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype = np.float32)
        self.reward = np.zeros((max_size, 1), dtype = np.float32)
        self.done = np.zeros((max_size, 1), dtype = np.float32)
        self.state_seq = np.zeros((max_size, c_step, state_dim), dtype = np.float32)  
        self.action_seq = np.zeros((max_size, c_step, 8), dtype = np.float32)  

    def add(self, state, action, reward, next_state, done, state_seq, action_seq):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = float(done)
        self.state_seq[self.ptr] = state_seq
        self.action_seq[self.ptr] = action_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = tf.random.uniform([batch_size,], 0, self.size, dtype=tf.int32)
        return (  
        tf.convert_to_tensor(self.state[ind]),
        tf.convert_to_tensor(self.action[ind]),
        tf.convert_to_tensor(self.next_state[ind]),
        tf.convert_to_tensor(self.reward[ind]),
        tf.convert_to_tensor(self.done[ind]),
        tf.convert_to_tensor(self.state_seq[ind]),
        tf.convert_to_tensor(self.action_seq[ind]))
 
