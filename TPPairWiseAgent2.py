# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021


import gym
import CIPairWiseEnv
from Config import Config
from stable_baselines.common.vec_env import DummyVecEnv

# stable-baselines implementation https://stable-baselines.readthedocs.io/en/master/index.html
# Tensorflow
class TPPairWiseAgent2:

    # Create the model
    def create_model(env, conf):
        # For each supported algorithm
        # Pairwise has discrete action space, so not all RL algorithm applies to the problem

        # Deep Q-Network https://stable-baselines.readthedocs.io/en/master/modules/dqn.html
        # DQN paper: https://arxiv.org/abs/1312.5602
        # Dueling DQN: https://arxiv.org/abs/1511.06581
        # Double-Q Learning: https://arxiv.org/abs/1509.06461
        # Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
        if conf.algo.upper() == "DQN":
            from stable_baselines import DQN
            from stable_baselines.deepq.policies import MlpPolicy
            model = DQN(MlpPolicy, env, gamma=conf.gamma, learning_rate=conf.learning_rate, buffer_size=conf.buffer_size,
                        exploration_fraction=conf.exploration_fraction, exploration_final_eps=conf.exploration_final_eps, exploration_initial_eps=conf.exploration_initial_eps,
                        train_freq=conf.train_freq, batch_size=conf.batch_size, double_q=True, learning_starts=conf.learning_starts,
                        target_network_update_freq=conf.target_network_update_freq, prioritized_replay=conf.prioritized_replay, prioritized_replay_alpha=conf.prioritized_replay_alpha,
                        prioritized_replay_beta0=conf.prioritized_replay_beta0, prioritized_replay_beta_iters=None,
                        prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0,
                        tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None,
                        full_tensorboard_log=False, seed=None)
            return model

        # A2C Advantage Actor Critic https://stable-baselines.readthedocs.io/en/master/modules/a2c.html
        # Original paper: https://arxiv.org/abs/1602.01783
        # OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/
        elif conf.algo.upper() == "A2C":
            env = DummyVecEnv([lambda: env])
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.a2c import A2C
            return A2C('MlpPolicy', env, gamma=conf.gamma, learning_rate=conf.learning_rate, verbose=conf.verbose,
                       tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None, seed=None)

        # PPO1 PPO2 Proximal Policy Optimization https://stable-baselines.readthedocs.io/en/master/modules/ppo1.html
        # Original paper: https://arxiv.org/abs/1707.06347
        # Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
        # PPO1 uses MPI for multiprocessing unlike PPO2, which uses vectorized environments.
        # PPO2 is the implementation OpenAI made for GPU.
        elif conf.algo.upper() == "PPO1":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.ppo1 import PPO1
            env = DummyVecEnv([lambda: env])
            return PPO1(MlpPolicy, env, verbose=conf.verbose, tensorboard_log=conf.tensorboard_log)

        elif conf.algo.upper() == "PPO2":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.ppo2 import PPO2
            env = DummyVecEnv([lambda: env])
            return PPO2('MlpPolicy', env, gamma=conf.gamma, learning_rate=conf.learning_rate, verbose=conf.verbose,
                        tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None, seed=None)

        # ACER Sample Efficient Actor-Critic with Experience Replay https://stable-baselines.readthedocs.io/en/master/modules/acer.html
        # Original paper: https://arxiv.org/abs/1611.01224
        elif conf.algo.upper() == "ACER":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.acer import ACER
            env = DummyVecEnv([lambda: env])
            return ACER(MlpPolicy, env, replay_ratio=conf.replay_ratio, verbose=conf.verbose, tensorboard_log=conf.tensorboard_log)

        # ACKTR Actor Critic using Kronecker-Factored Trust Region
        # https://stable-baselines.readthedocs.io/en/master/modules/acktr.html
        # Original paper: https://arxiv.org/abs/1708.05144
        # Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
        elif conf.algo.upper() == "ACKTR":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.acktr import ACKTR
            env = DummyVecEnv([lambda: env])
            return ACKTR(MlpPolicy, env, verbose=conf.verbose, tensorboard_log=conf.tensorboard_log)

        # TRPO Trust Region Policy Optimization https://stable-baselines.readthedocs.io/en/master/modules/trpo.html
        # Original paper: https://arxiv.org/abs/1502.05477
        # OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
        elif conf.algo.upper() == "TRPO":
            from stable_baselines.common.policies import MlpPolicy
            from stable_baselines.trpo_mpi import TRPO
            env = DummyVecEnv([lambda: env])
            return TRPO(MlpPolicy, env, verbose=conf.verbose, tensorboard_log=conf.tensorboard_log)

        else:
            return None

    # Load Model from disk
    def load_model(env, algo, path):
        if algo.upper() == "A2C":
            from stable_baselines.a2c import A2C
            model = A2C.load(path)
            env = DummyVecEnv([lambda: env])
        elif algo.upper() == "DQN":
            from stable_baselines.deepq import DQN
            model = DQN.load(path)
        elif algo.upper() == "PPO2":
            from stable_baselines.ppo2 import PPO2
            model = PPO2.load(path)
            env = DummyVecEnv([lambda: env])
        elif algo.upper() == "ACER":
            from stable_baselines.acer import ACER
            model = ACER.load(path)
            env = DummyVecEnv([lambda: env])
        elif algo.upper() == "ACKTR":
            from stable_baselines.acktr import ACKTR
            model = ACKTR.load(path)
            env = DummyVecEnv([lambda: env])
        elif algo.upper() == "PPO1":
            from stable_baselines.ppo1 import PPO1
            model = PPO1.load(path)
            env = DummyVecEnv([lambda: env])
        elif algo.upper() == "TRPO":
            from stable_baselines.trpo_mpi import TRPO
            model = TRPO.load(path)
            env = DummyVecEnv([lambda: env])
        else:
            return None

        model.set_env(env)
        return model

    # Train the model, returns the trained model
    def train_agent(env: CIPairWiseEnv, conf: Config, path_to_save_agent: None, base_model=None,  callback_class=None):
        # Reset environment
        env.reset()

        # Get model, if no model defined create one
        if not base_model:
            base_model = TPPairWiseAgent2.create_model(env,conf)
        else:
            env = DummyVecEnv([lambda: env])
            base_model.set_env(env)

        # Learn
        base_model = base_model.learn(total_timesteps=conf.steps, reset_num_timesteps=False)

        # Save model
        if path_to_save_agent:
            base_model.save(path_to_save_agent)

        return base_model

    # Test model, returns the predicted order of test cases
    def test_agent(env: CIPairWiseEnv, model_path: str, model):
        agent_actions = []

        # If no model passed, load model
        if not model:
            model = TPPairWiseAgent2.load_model(env, model_path)

        if model:
            env = DummyVecEnv([lambda: env])

            model.set_env(env)
            obs = env.reset()

            env.get_attr("test_cases_vector")

            done = False
            while True:
                # Get action
                action, _states = model.predict(obs, deterministic=False)

                # Step
                obs, rewards, done, info = env.step(action)

                # Done
                if done:
                    break

            return env.get_attr("sorted_test_cases_vector")[0]