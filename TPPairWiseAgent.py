# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021


import gym
import CIPairWiseEnv
from Config import Config
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# stable-baselines3 implementation https://stable-baselines3.readthedocs.io/en/master/#
# Pytorch

class TPPairWiseAgent:

    # Create the model
    def create_model(env, conf):
        # A2C Advantage Actor Critic https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
        # Original paper: https://arxiv.org/abs/1602.01783
        # OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/
        if conf.algo.upper() == "A2C":
            return A2C('MlpPolicy', env, gamma=conf.gamma, learning_rate=conf.learning_rate, verbose=conf.verbose,
                       tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None, seed=None)

        # PPO Proximal Policy Optimization https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        # Original paper: https://arxiv.org/abs/1707.06347
        # Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
        # OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
        # Spinning Up guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        elif conf.algo.upper() == "PPO":
            return PPO('MlpPolicy', env, gamma=conf.gamma, learning_rate=conf.learning_rate, verbose=conf.verbose,
                       tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None, seed=None)

        # Deep Q Network https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
        # Original paper: https://arxiv.org/abs/1312.5602
        # Further reference: https://www.nature.com/articles/nature14236
        elif conf.algo.upper() == "DQN":
            return DQN('MlpPolicy', env, gamma=conf.gamma, learning_rate=conf.learning_rate, verbose=conf.verbose,
                       tensorboard_log=conf.tensorboard_log, _init_setup_model=True, policy_kwargs=None, seed=None)

    # Load the model from disk
    def load_model(env, algo, path):
        if algo.upper() == "A2C":
            model = A2C.load(path)
        elif algo.upper() == "DQN":
            model = DQN.load(path)
        elif algo.upper() == "PPO":
            model = PPO.load(path)
        model.set_env(env)
        return model

    # Train the model, returns trained model
    def train_agent(env: CIPairWiseEnv, conf: Config, path_to_save_agent: None, base_model=None,  callback_class=None):

        # Reset environment
        env.reset()

        # Get model
        if not base_model:
            base_model = TPPairWiseAgent.create_model(env,conf)
        else:
            env = DummyVecEnv([lambda: env])
            base_model.set_env(env)

        # Learn
        base_model = base_model.learn(total_timesteps=conf.steps, reset_num_timesteps=False, callback=callback_class)

        # Save model
        if path_to_save_agent:
            base_model.save(path_to_save_agent)

        return base_model

    # Test model, returns the predicted order of test cases
    def test_agent(env: CIPairWiseEnv, model_path: str, model):
        agent_actions = []

        if not model:
            model = TPPairWiseAgent.load_model(env, model_path)

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