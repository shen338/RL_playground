import sys
sys.path.append("..")
from utils import logz
import tensorflow as tf 
import numpy as np
import time 
import inspect
import gym
import uuid
from agents.DQN_utils import *
import pickle
from collections import namedtuple

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


class DQNAgent(object): 

    def __init__(
        self,
        env,
        q_func,
        optimizer_spec,
        session,
        exploration=LinearSchedule(1000000, 0.1),
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        rew_file=None,
        double_q=False):
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using q_func.

        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_func: function
            Model to use for computing the q function. It should accept the
            following named arguments:
                img_in: tf.Tensor
                    tensorflow tensor representing the input image
                num_actions: int
                    number of actions
                scope: str
                    scope in which all the model related variables
                    should be created
                reuse: bool
                    whether previously created variables should be reused.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        session: tf.Session
            tensorflow session to use.
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        stopping_criterion: (env, t) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        double_q: bool
            If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
            https://papers.nips.cc/paper/3964-double-q-learning.pdf
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space)      == gym.spaces.Discrete

        self.target_update_freq = target_update_freq
        self.optimizer_spec = optimizer_spec
        self.batch_size = batch_size
        self.learning_freq = learning_freq
        self.learning_starts = learning_starts
        self.stopping_criterion = stopping_criterion
        self.env = env
        self.session = session
        self.exploration = exploration
        self.rew_file = str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file
        self.double_q = double_q

        ###############
        # BUILD MODEL #
        ###############

        if len(self.env.observation_space.shape) == 1:
            # This means we are running on low-dimensional observations (e.g. RAM)
            input_shape = self.env.observation_space.shape
        else:
            img_h, img_w, img_c = self.env.observation_space.shape
            input_shape = (img_h, img_w, frame_history_len * img_c)
        self.num_actions = self.env.action_space.n

        # set up placeholders
        # placeholder for current observation (or state)
        self.obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
        # placeholder for current action
        self.act_t_ph              = tf.placeholder(tf.int32,   [None])
        # placeholder for current reward
        self.rew_t_ph              = tf.placeholder(tf.float32, [None])
        # placeholder for next observation (or state)
        self.obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
        # placeholder for end of episode mask
        # this value is 1 if the next state corresponds to the end of an episode,
        # in which case there is no Q-value at the next state; at the end of an
        # episode, only the current state reward contributes to the target, not the
        # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
        self.done_mask_ph          = tf.placeholder(tf.float32, [None])
        
        obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
        obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

        # compute the Bellman error. This requires
        # evaluating the current and next Q-values and constructing the corresponding error.
        # TensorFlow will differentiate this error for you, you just need to pass it to the
        # optimizer. See assignment text for details.
        # Your code should produce one scalar-valued tensor: total_error
        # This will be passed to the optimizer in the provided code below.
        # Your code should also produce two collections of variables:
        # q_func_vars
        # target_q_func_vars
        # These should hold all of the variables of the Q-function network and target network,
        # respectively. A convenient way to get these is to make use of TF's "scope" feature.
        # For example, you can create your Q-function network with the scope "q_func" like this:
        # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
        # And then you can obtain the variables like this:
        # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
        # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
        ######

        # Compute Bellman Error
        self.q_all_action = q_func(obs_t_float, self.num_actions, scope="q_func", reuse=False) 
        self.target_all_action = q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)
        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        # if using Double Q learning, the best action is determined by inner network but the Q_next is determined by target network
        # if not using Double Q learning, the best action and Q_next is determined by target network 

        q_next = tf.reduce_max(self.target_all_action, axis=1)

        if self.double_q:
            best_next_action = tf.math.argmax(self.q_all_action, axis=1)
            best_next_action = tf.one_hot(best_next_action, depth=self.num_actions)
            q_next = tf.reduce_sum(self.target_all_action*best_next_action, axis=1)

        target_q_func = gamma*q_next*(1 - self.done_mask_ph) + self.rew_t_ph
        
        # Huber Loss
        # Slice corresponding action in q_all_action
        q_current_action = tf.reduce_sum(self.q_all_action * tf.one_hot(self.act_t_ph, depth=self.num_actions), axis=1)
        self.total_error = tf.losses.huber_loss(target_q_func, q_current_action)

        ######
        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                    var_list=q_func_vars, clip_val=grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

        # construct the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)
        self.replay_buffer_idx = None

        ###############
        # RUN ENV     #
        ###############
        self.model_initialized = False
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = self.env.reset()
        self.log_every_n_steps = 10000

        self.start_time = None
        self.t = 0

    def stopping_criterion_met(self):
        return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

    def step_env(self):

        ### 2. Step the env and store the transition
        # At this point, "self.last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, self.last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "self.last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####

        # YOUR CODE HERE
        # one step of env execution from self.last_obs
        buffer_index = self.replay_buffer.store_frame(self.last_obs) 
        # find next action with episilon exploration 
        if np.random.random() < self.exploration.value(self.t) or (not self.model_initialized):
            current_act = np.random.randint(0, self.num_actions)
        else:
            current_obs = self.replay_buffer.encode_recent_observation()
            q_all_action = self.session.run(self.q_all_action, feed_dict={self.obs_t_ph: current_obs[None, :]})
            current_act = np.argmax(q_all_action, axis=1)
        
        # execute one step 
        next_obs, current_reward, done, _ = self.env.step(current_act)
        self.replay_buffer.store_effect(buffer_index, current_act, current_reward, done)
        if done:
            self.last_obs = self.env.reset()
        else: 
            self.last_obs = next_obs

    def update_model(self):

        if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):

            obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(self.batch_size)

            if not self.model_initialized: 
                initialize_interdependent_variables(self.session, tf.global_variables(), {
                    self.obs_t_ph: obs_t_batch,
                    self.obs_tp1_ph: obs_tp1_batch
                })
                self.session.run(self.update_target_fn)
                self.model_initialized = True
                
            _, losses = self.session.run([self.train_fn, self.total_error], feed_dict={self.obs_t_ph: obs_t_batch, 
                                                            self.act_t_ph: act_t_batch,
                                                            self.rew_t_ph: rew_t_batch,
                                                            self.obs_tp1_ph: obs_tp1_batch,
                                                            self.done_mask_ph: done_mask, 
                                                            self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)})

            if self.t % self.target_update_freq == 0:
                # just copy var from q_func to target_q_func
                self.session.run([self.update_target_fn])
                self.num_param_updates += 1

        self.t += 1

    def log_progress(self):
        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])

        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        if self.t % self.log_every_n_steps == 0 and self.model_initialized:
            print("Timestep %d" % (self.t,))
            print("mean reward (100 episodes) %f" % self.mean_episode_reward)
            print("best mean reward %f" % self.best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % self.exploration.value(self.t))
            print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
            if self.start_time is not None and self.t % self.log_every_n_steps == 0:
                print("running time %f" % ((time.time() - self.start_time) / 60.))

            self.start_time = time.time()

            sys.stdout.flush()

            with open(self.rew_file, 'wb') as f:
                pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def DQN_train(*args, **kwargs):
    alg = DQNAgent(*args, **kwargs)

    while not alg.stopping_criterion_met():
        alg.step_env()
        alg.update_model()
        alg.log_progress()