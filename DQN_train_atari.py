import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import agents.DQN as dqn
from agents.DQN_utils import *
from utils.atari_wrappers import *
from network.body import SmallConvBody

def atari_learn(env,
                session,
                num_timesteps, 
                replay_buffer_size=1000000,
                batch_size=32,
                gamma=0.99,
                learning_freq=4,
                frame_history_len=4,
                target_update_freq=10000,
                grad_norm_clipping=10,
                double_q=True):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.DQN_train(
        env=env,
        q_func=SmallConvBody,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        learning_starts=50000,
        learning_freq=learning_freq,
        frame_history_len=frame_history_len,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=True
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env = gym.make('PongNoFrameskip-v4')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gamma', type=int, default=0.99)
    parser.add_argument('--learning_freq', type=int, default=4)
    parser.add_argument('--frame_history_len', type=int, default=4)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument("--double_q", type=bool, default=False, help="Activate nice mode.")
    args = parser.parse_args()

    # Get Atari games.
    task = gym.make('PongNoFrameskip-v4')

    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, num_timesteps=2e8, replay_buffer_size=args.replay_buffer_size,
                batch_size=args.batch_size,
                gamma=args.gamma,
                learning_freq=args.learning_freq,
                frame_history_len=args.frame_history_len,
                target_update_freq=args.target_update_freq,
                double_q=args.double_q)

if __name__ == "__main__":
    main()
