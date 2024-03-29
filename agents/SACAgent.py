import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_probability import distributions
from tensorflow.python import keras
from tensorflow.python.keras.engine.network import Network
import time

# Python implementation of Soft Actor-Critic (SAC) from HW5 of CS294-112

class QFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = [
            layers.Input(batch_shape=input_shape[0], name='observations'),
            layers.Input(batch_shape=input_shape[1], name='actions')
        ]

        x = layers.Concatenate(axis=1)(inputs)
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        q_values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, q_values)
        super(QFunction, self).build(input_shape)


class ValueFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, values)
        super(ValueFunction, self).build(input_shape)


class GaussianPolicy(Network):
    def __init__(self, action_dim, hidden_layer_sizes, reparameterize, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self._action_dim = action_dim
        self._f = None
        self._hidden_layer_sizes = hidden_layer_sizes
        self._reparameterize = reparameterize

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)

        mean_and_log_std = layers.Dense(
            self._action_dim * 2, activation=None)(x)

        def create_distribution_layer(mean_and_log_std):
            mean, log_std = tf.split(
                mean_and_log_std, num_or_size_splits=2, axis=1)
            log_std = tf.clip_by_value(log_std, -20., 2.)

            distribution = distributions.MultivariateNormalDiag(
                loc=mean,
                scale_diag=tf.exp(log_std))

            raw_actions = distribution.sample()
            if not self._reparameterize:
                ### Problem 1.3.A
                ### YOUR CODE HERE
                raw_actions = tf.stop_gradient(raw_actions)
            log_probs = distribution.log_prob(raw_actions)
            log_probs -= self._squash_correction(raw_actions)

            actions = None
            ### Problem 2.A
            ### YOUR CODE HERE
            actions = tf.tanh(raw_actions)

            return actions, log_probs

        samples, log_probs = layers.Lambda(create_distribution_layer)(
            mean_and_log_std)

        self._init_graph_network(inputs=inputs, outputs=[samples, log_probs])
        super(GaussianPolicy, self).build(input_shape)

    def _squash_correction(self, raw_actions, eps=1e-8):
        ### Problem 2.B
        ### YOUR CODE HERE
        # raise NotImplementedError
        true_action = tf.reduce_sum(tf.log(4) -
                                 2 * (tf.nn.softplus(2 * raw_actions) - raw_actions), axis=1)
       
        return true_action

    def eval(self, observation):
        assert self.built and observation.ndim == 1

        if self._f is None:
            self._f = keras.backend.function(self.inputs, [self.outputs[0]])

        action, = self._f([observation[None]])
        return action.flatten()


class SACAgent(object):
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 **kwargs):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self._training_ops = []

    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function):

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for(policy, q_function, q_function2, value_function)
        value_function_loss = self._value_function_loss_for(
            policy, q_function, q_function2, value_function)
        q_function_loss = self._q_function_loss_for(q_function,
                                                    target_value_function)
        if q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(q_function2,
                                                        target_value_function)

        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=q_function.trainable_variables)
        if q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=q_function2.trainable_variables)

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        if q_function2 is not None:
            self._training_ops += [q_function2_training_op]
        self._target_update_ops = self._create_target_update(
            source=value_function, target=target_value_function)

        tf.get_default_session().run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    def _policy_loss_for(self, policy, q_function, q_function2, value_function):
        if not self._reparameterize:
            ### Problem 1.3.A
            ### YOUR CODE HERE
            actions, log_pis = policy(self._observations_ph)
            if q_function2 is None:
                q_values = tf.squeeze(q_function((self._observations_ph, self._actions_ph)), axis=1)
            else: 
                q_values = tf.squeeze(tf.minimum(q_function((self._observations_ph, actions)),
                                        q_function2((self._observations_ph, actions))), axis=1)
            # Baseline value functions
            baseline = tf.squeeze(value_function(self._observations_ph), axis=1)
            # REINFORCE style policy update 
            policy_loss = tf.reduce_mean(log_pis * (tf.stop_gradient(self._alpha*log_pis - q_values + baseline)))
        else:
            ### Problem 1.3.B
            ### YOUR CODE HERE
            actions, log_pis = policy(self._observations_ph)
            if q_function2 is None:
                q_values = tf.squeeze(q_function((self._observations_ph, self._actions_ph)), axis=1)
            else: 
                q_values = tf.squeeze(tf.minimum(q_function((self._observations_ph, actions)),
                                        q_function2((self._observations_ph, actions))), axis=1)
            # Baseline value functions
            # baseline = tf.squeeze(value_function(self._observations_ph), axis=1)
            targets = self._alpha * log_pis - q_values
            policy_loss = tf.reduce_mean(targets)
            
        return policy_loss

    def _value_function_loss_for(self, policy, q_function, q_function2, value_function):
        ### Problem 1.2.A
        ### YOUR CODE HERE
        
        actions, log_pis = policy(self._observations_ph)
        values = tf.squeeze(value_function(self._observations_ph), axis=1)
        if q_function2 is None:
            q_values = tf.squeeze(q_function((self._observations_ph, self._actions_ph)), axis=1)
        else: 
            q_values = tf.squeeze(tf.minimum(q_function((self._observations_ph, actions)),
                                    q_function2((self._observations_ph, actions))), axis=1)

        target = - self._alpha*log_pis + q_values
        loss = tf.losses.mean_squared_error(target, values)

        return loss
        
    def _q_function_loss_for(self, q_function, target_value_function):
        ### Problem 1.1.A
        ### YOUR CODE HERE
        q_values = tf.squeeze(q_function((self._observations_ph, self._actions_ph)), axis=1)
        values = tf.squeeze(target_value_function(self._next_observations_ph), axis=1)
        target_q_values = values*(1-self._terminals_ph)*self._discount + self._rewards_ph
        loss = tf.losses.mean_squared_error(target_q_values, q_values)

        return loss

    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                tf.get_default_session().run(self._training_ops, feed_dict)
                tf.get_default_session().run(self._target_update_ops)

            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }

        return statistics
