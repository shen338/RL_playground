import sys
sys.path.append("..")
from network.body import MLPBody
from utils import logz
import tensorflow as tf 
import numpy as np
import time 
import inspect
import gym


## implement GAE on the basis of Actor Critic
class ACAgent(object):
    
    def __init__(self, computation_args, env_args, sample_traj_args, estimate_return_args):
        super().__init__()

        self.ob_dim = env_args['ob_dim']
        self.ac_dim = env_args['ac_dim']
        self.discrete = env_args['discrete']
        self.size = computation_args['size']
        self.n_layers = computation_args['n_layers']
        self.learning_rate = computation_args['learning_rate']
        self.num_target_updates = computation_args['num_target_updates']
        self.num_grad_steps_per_target_update = computation_args['num_grad_steps_per_target_update']

        self.animate = sample_traj_args['animate']
        self.max_path_length = sample_traj_args['max_path_length']
        self.min_timesteps_per_batch = sample_traj_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.lamb = estimate_return_args['lambda']
        self.normalize_advantages = estimate_return_args['normalize_advantages']

    def init_tf_sess(self):
        
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
        tf_config.gpu_options.allow_growth = True
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

        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            # if discrete, only get a vector of actions to choose 
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 

        sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)

        return sy_ob_no, sy_ac_na, sy_adv_n
    
    def policy_forward_pass(self, sy_ob_no):
        
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean_na: (batch_size, self.ac_dim)
                    sy_logstd_a: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """

        if self.discrete:
            MLP = MLPBody(sy_ob_no, self.ac_dim, "MLP", self.n_layers, self.size, output_activation=None)
            sy_logits_na = MLP.outputs
            return sy_logits_na
        else:
            MLP = MLPBody(sy_ob_no, self.ac_dim, "MLP", self.n_layers, self.size, output_activation=None)
            sy_mean_na = MLP.outputs
            sy_logstd_a = tf.Variable(tf.zeros([self.ac_dim]), name="policy/logstd", dtype=tf.float32)
            return (sy_mean_na, sy_logstd_a)

    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean_na: (batch_size, self.ac_dim)
                        sy_logstd_a: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)
        """
        
        if self.discrete:
            sy_logits_na = policy_parameters
            sy_sampled_ac = tf.multinomial(sy_logits_na, 1)
            sy_sampled_ac = tf.reshape(sy_sampled_ac, [-1])
        else:
            sy_mean_na, sy_logstd_a = policy_parameters
            random_number = tf.random_normal(tf.shape(sy_mean_na))
            sy_std = tf.exp(sy_logstd_a)
            sy_sampled_ac = random_number*sy_std + sy_mean_na
            sy_sampled_ac = tf.reshape(sy_sampled_ac, [-1])
        return sy_sampled_ac

    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """

        if self.discrete:
            sy_logits_na = policy_parameters
            # One-hot plus logrithm 
            sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sy_logits_na, labels=sy_ac_na)
        else:
            # Calculate Gaussian Prob
            sy_mean, sy_logstd = policy_parameters
            sy_std = tf.exp(sy_logstd)
            sy_z = (sy_ac_na - sy_mean) / sy_std
            sy_logprob_n = -0.5 * tf.reduce_sum(tf.square(sy_z), axis=1)
        return sy_logprob_n

    def build_computation_graph(self):

        """Run the computation graph, get loss, define Optimizer, get policy gradient and critic loss
        """

        (self.sy_ob_no, self.sy_ac_na, self.sy_adv_n) = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        # Loss function 
        actor_loss = - tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)
        self.actor_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(actor_loss)

        # Define critic 
        NN_critic = MLPBody(self.sy_ob_no,
                                1,
                                "nn_critic",
                                n_layers=self.n_layers,
                                size=self.size)
        self.critic_prediction = tf.squeeze(NN_critic.outputs)
        self.sy_target_n = tf.placeholder(shape=[None], name="critic_target", dtype=tf.float32)
        self.critic_loss = tf.losses.mean_squared_error(self.sy_target_n, self.critic_prediction)
        self.critic_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

    def sample_trajs(self, itr, env): 

        """sample trajectories to feed into one training batch 
        """

        timesteps_this_batch = 0
        paths = []
        # start_time = time.time()

        while True:
            animate_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_traj(env, animate_episode)
            paths.append(path)
            timesteps_this_batch += len(path["reward"])
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break

        # Implementation of Multi Threading
        # from multiprocessing.dummy import Pool as ThreadPool 
        # pool = ThreadPool(8) 
        # while True:
        #     envs = [env]*8
        #     results = pool.map(self.sample_trajectory, envs)
        #     for res in results: 
        #         timesteps_this_batch += pathlength(res)
        #         paths.append(res)
        #     if timesteps_this_batch > self.min_timesteps_per_batch:
        #         break

        # print(time.time() - start_time)


        # Implementation of Multiprocessing Trajectory Sampling
        # from multiprocessing import Pool, cpu_count, Process
        # pool = Pool()
        # while True:
            
        #     cpus = cpu_count()
        #     print(cpus)
        #     process = []
        #     for ii in range(4):
        #         p = Process(target=self.sample_trajectory, args=(env,))
        #         process.append(p)

        #     for ii in range(4):
        #         i = process[ii].start()
        #         print(i)
        #         timesteps_this_batch = pathlength(i)
        #         paths.append(i)

        return paths, timesteps_this_batch

    def sample_traj(self, env, animate_episode=True):
        
        ob = env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            if animate_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None, :]})
            ac = ac[0]
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            # add the observation after taking a step to next_obs
            next_obs.append(ob)
            rewards.append(rew)
            steps += 1
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            if done or steps > self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32),
                "next_observation": np.array(next_obs, dtype=np.float32),
                "terminal": np.array(terminals, dtype=np.float32)}
        return path

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):

        """Core code for GAE: 
        \widehat{A}_{t}^{G A E(\gamma, \lambda)}=\sum_{l=1}^{\infty}(\gamma \lambda)^{l} \delta_{t+l}^{V}=\sum_{l=1}^{\infty}(\gamma \lambda)^{l}\left(r_{t}+\gamma V\left(s_{t+l+1}\right)-V\left(s_{t+l}\right)\right)
        """
        value_s = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: ob_no})
        value_s_next = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
        
        adv_n = []
        start = 0
        for path in re_n:
            length = len(path)
            value_s_path = value_s[start:start+length]
            value_s_next_path = value_s_next[start:start+length]
            value_s_next_path[-1] = 0  # For last step in a trajectory, terminal_n = 1
            adv_path = []
            delta = [self.gamma*value_s_next_path[ii] - value_s_path[ii] + path[ii] for ii in range(length)]
            # delta = [v_path[ii+1] + path[ii] - v_path[ii] for ii in range(length)]
            cumulative = 0
            for item in reversed(delta):
                cumulative = item + cumulative*self.gamma*self.lamb
                adv_path.append(cumulative)
            adv_path.reverse()
            adv_n.extend(adv_path)
            start += length

        return adv_n

    def update_critic(self, ob_no, next_ob_no, re_n, terminal_n):

        for _ in range(self.num_grad_steps_per_target_update):
            value_s_prime = self.sess.run(self.critic_prediction, feed_dict={self.sy_ob_no: next_ob_no})
            Q_sa = re_n + self.gamma*value_s_prime*(1-terminal_n)
            for _ in range(self.num_target_updates):
                self.sess.run(self.critic_update_op,
                    feed_dict={self.sy_target_n: Q_sa, self.sy_ob_no: ob_no})

    def update_actor(self, ob_no, ac_na, adv_n):
        
        self.sess.run(self.actor_update_op, feed_dict={self.sy_ob_no: ob_no,
                                                        self.sy_ac_na: ac_na, 
                                                        self.sy_adv_n: adv_n})

def AC_train(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        lamb,
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate,
        num_target_updates,
        num_grad_steps_per_target_update,
        animate, 
        logdir, 
        normalize_advantages,
        seed,
        n_layers,
        size):
    
    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    # args = inspect.getargspec(PG_train)[0]
    # params = {k: locals()[k] if k in locals() else None for k in args}
    params = locals()
    print(params)
    logz.save_params(params)

    # Make the gym environment
    env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    # initialize Policy Gradient Agent
    network_args = {
        'n_layers': n_layers, 
        'size':size, 
        'learning_rate': learning_rate,
        'num_target_updates': num_target_updates,
        'num_grad_steps_per_target_update': num_grad_steps_per_target_update
    }
    env_args = {
        'ob_dim': ob_dim,
        'ac_dim': ac_dim, 
        'discrete': discrete, 
    }
    sample_traj_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }
    estimate_return_args = {
        'gamma': gamma,
        'lambda': lamb,
        'normalize_advantages': normalize_advantages,
    }

    # Agent
    agent = ACAgent(network_args, env_args, sample_traj_args, estimate_return_args)

    agent.build_computation_graph()
    agent.init_tf_sess()

    # start training 
    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajs(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [[path["reward"] for path in paths]]
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])
        
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        re_n = np.concatenate(re_n)
        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, ac_na, adv_n)
        
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [len(path["reward"]) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()