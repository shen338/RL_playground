import sys
sys.path.append("..")
from network.body import MLPBody
from utils import logz
import tensorflow as tf 
import numpy as np
import time 
import inspect
import gym


class ReplayBuffer(object):
    # A simple FIFO replay buffer, only need store_batch and sample_batch function 
    # adopted from https://github.com/openai/spinningup/blob/master/spinup/algos/ddpg/ddpg.py

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def actor_critic(ob, ac, hidden_dim, scope, action_space):
    # define critic network and actor network for DDPG 
    ac_dim = ac.shape.as_list()[-1]
    ac_limit = action_space.high[0]
    with tf.variable_scope(scope):

        actor = MLPBody(ob, ac_dim, scope="actor", n_layers=2, size=500, activation=tf.nn.relu, output_activation=tf.nn.tanh)
        sample_a = actor.outputs * ac_limit

        critic = MLPBody(tf.concat([ob, ac], axis=1), 1, scope="critic", n_layers=2, size=500, activation=tf.nn.relu, output_activation=None)

        critic_pi = MLPBody(tf.concat([ob, sample_a], axis=1), 1, scope="critic", n_layers=2, size=500, activation=tf.nn.relu, output_activation=None, reuse=True)

    return sample_a, tf.squeeze(critic.outputs, axis=1), tf.squeeze(critic_pi.outputs, axis=1)

class DDPGAgent(object):

    def __init__(self, env, actor_critic=actor_critic, gamma=0.99, 
         polyak=0.995, actor_lr=1e-3, critic_lr=1e-3, act_noise=0.1):

        super().__init__()

        self.env = env

        assert isinstance(env.action_space, gym.Space.Box)  # make sure action space is continuous
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.ac_limit = env.action_space.high[0]

        self.actor_critic = actor_critic

        self.gamma = gamma
        self.polyak = polyak

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.act_noise = act_noise

        self.init_tf_sess()

        self.build_computation_graph()

    def init_tf_sess(self):
        
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    def define_placeholders(self):
    
        """Place Holder for batch observations, actions, advantages
        
            returns: 
                sy_ob_no: placeholder for observation
                sy_ac_na: placeholder for action
                sy_adv_n: placeholder for advantage 
        """

        ph_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        ph_ob_next_no = tf.placeholder(shape=[None, self.ob_dim], name="ob_next", dtype=tf.float32)

        ph_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 

        ph_rew_n = tf.placeholder(shape=[None, ], name="rew", dtype=tf.float32)

        ph_done_n = tf.placeholder(shape=[None, ], name="done_flag", dtype=tf.int32)

        return ph_ob_no, ph_ob_next_no, ph_ac_na, ph_rew_n, ph_done_n

    def sample_action(self, ob):

        ac = self.sess.run(self.target_sample_a, feed_dict={self.ph_ob_next_no: ob})
        ac += self.act_noise * np.random.randn(self.ac_dim)
        return np.clip(ac, -self.ac_limit, self.ac_limit)

    def build_computation_graph(self):

        # get placeholders first 
        self.ph_ob_no, self.ph_ob_next_no, self.ph_ac_na, self.ph_rew_n, self.ph_done_n = self.define_placeholders()
        
        # get critic network and actor network 
        self.sample_a, critic, critic_pi = actor_critic(self.ph_ob_no, self.ph_ac_na, hidden_dim=500, scope="main", action_space=self.env.action_space)

        # get target critic network and actor network 
        self.target_sample_a, _, target_critic_pi = actor_critic(self.ph_ob_no, self.ph_ac_na, hidden_dim=500, scope="target", action_space=self.env.action_space)

        target_q = tf.stop_gradient(self.ph_rew_n + self.gamma*target_critic_pi*(1-self.ph_done_n))

        self.critic_loss = tf.losses.mean_squared_error(target_q, critic)

        self.actor_loss = tf.reduce_mean(critic_pi) 

        self.critic_update_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss, var_list=[x for x in tf.global_variables() if "main/critic" in x.name])

        self.actor_update_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss, var_list=[x for x in tf.global_variables() if "main/actor" in x.name])

        # Polyak averaging for target variables
        self.target_update_op = tf.group([tf.assign(v_targ, self.polyak*v_targ + (1-self.polyak)*v_main)
                                for v_main, v_targ in zip([x for x in tf.global_variables() if "main" in x.name], [x for x in tf.global_variables() if "target" in x.name])])

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip([x for x in tf.global_variables() if "main" in x.name], [x for x in tf.global_variables() if "target" in x.name])])

    def update_actor_and_target(self, ob_no, ob_next_no, ac_na, rew_n, done_n): 

        actor_loss, _, _ = self.sess.run([self.actor_loss, self.actor_update_op, self.target_update_op], feed_dict = {self.ph_ob_no:ob_no, 
                                                                                            self.ph_ob_next_no: ob_next_no,
                                                                                            self.ph_ac_na: ac_na, 
                                                                                            self.ph_rew_n: rew_n, 
                                                                                            self.ph_done_n: done_n})

        return actor_loss
    
    def update_critic(self, ob_no, ob_next_no, ac_na, rew_n, done_n):
        
        critic_loss, _ = self.sess.run([self.critic_loss, self.critic_update_op], feed_dict = {self.ph_ob_no:ob_no, 
                                                                                            self.ph_ob_next_no: ob_next_no,
                                                                                            self.ph_ac_na: ac_na, 
                                                                                            self.ph_rew_n: rew_n, 
                                                                                            self.ph_done_n: done_n})

        return critic_loss                                            

def DDPG_train(env, logdir='.', actor_critic=actor_critic, iterations=600000, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, actor_lr=1e-3, critic_lr=1e-3, batch_size=100, start_steps=10000, act_noise=0.1):

    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    # args = inspect.getargspec(PG_train)[0]
    # params = {k: locals()[k] if k in locals() else None for k in args}
    params = locals()
    print(params)
    logz.save_params(params)
    
    ddpg = DDPGAgent(env, actor_critic, gamma, polyak, actor_lr, critic_lr, act_noise)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(ob_dim, ac_dim, replay_size)

    start_time = time.time()
    ob = env.reset()
    ac, rew, done = 0, 0, 0
    actor_loss = []
    critic_loss = []
    
    for ii in range(iterations):

        if ii < start_steps: 
            ac = env.action_space.sample()
        else:
            ac = ddpg.sample_action(ob)

        ob_next, rew, done = env.step(ac)

        replay_buffer.store(ob, ac, rew, ob_next, done)

        if done is True:
            ob = env.reset() 

        # if iteration < start_step, only put steps into buffer
        if ii < start_steps: 
            continue

        # DDPG update 
        batch = replay_buffer.sample_batch(batch_size=batch_size)

        # update critic 
        a_loss = ddpg.update_critic(batch['obs1'], batch['obs2'], batch['acts'], batch['rews'], batch['done'])
        actor_loss.append(a_loss)

        # update actor and target
        c_loss = ddpg.update_actor_and_target(batch['obs1'], batch['obs2'], batch['acts'], batch['rews'], batch['done'])
        critic_loss.append(c_loss)

        if ii % 10000 == 0: 
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", ii)
            logz.log_tabular("AverageActorLoss", np.mean(np.array(actor_loss)))
            logz.log_tabular("AverageCriticLoss", np.mean(np.array(critic_loss)))
            logz.log_tabular("AverageActorStd", np.std(np.array(actor_loss)))
            logz.log_tabular("AverageCriticStd", np.std(np.array(critic_loss)))
            logz.dump_tabular()
            logz.pickle_tf_vars()


        

            



        


   


