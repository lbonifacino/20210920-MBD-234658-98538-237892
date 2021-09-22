# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021

import gym
import random
import numpy as np

from CICycleLog import CICycleLog
from Config import Config
from sklearn import preprocessing
from gym import spaces

# RL Environment
# The environment implements Mergesort to order the test cases of a CI Cycle.
class CIPairWiseEnv(gym.Env):
    def __init__(self, cycle_logs: CICycleLog, conf: Config):
        super(CIPairWiseEnv, self).__init__()

        self.conf = conf
        self.cycle_logs = cycle_logs
        self.calculate_reward = conf.calculate_reward

        random.shuffle(self.cycle_logs.test_cases)

        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0

        self.testcase_vector_size = self.cycle_logs.get_test_case_vector_length(cycle_logs.test_cases[0],
                                                                                self.conf.win_size)
        self.initial_observation = cycle_logs.test_cases.copy()
        self.test_cases_vector = self.initial_observation.copy()
        self.test_cases_vector_temp = []

        self.current_indexes = [0, 1]
        self.current_indexes[0] = self.index
        self.current_indexes[1] = self.index + self.width

        self.current_obs = self.get_pair_data(self.current_indexes)
        self.current_obs = self.get_pair_data(self.current_indexes)

        self.number_of_actions = 2
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, self.testcase_vector_size))

    # Obtain pair of test cases to sort
    def get_pair_data(self, current_indexes):
        i = 0
        test_case_vector_length = self.cycle_logs.get_test_case_vector_length(
            self.test_cases_vector[current_indexes[0]], self.conf.win_size)
        temp_obs = np.zeros((2, test_case_vector_length))

        for test_index in current_indexes:
            # ToDo
            #  Analyze other forms to enrich test case info
            temp_obs[i, :] = self.cycle_logs.export_test_case(self.test_cases_vector[test_index],
                                                              "list_avg_exec_with_failed_history",
                                                              self.conf.padding_digit,
                                                              self.conf.win_size)
            i = i + 1

        # Normalize values to make learning easier
        temp_obs = preprocessing.normalize(temp_obs, axis=0, norm='max')
        return temp_obs

    def render(self, mode='human'):
        pass

    # Reset environment
    def reset(self):
        self.test_cases_vector = self.initial_observation.copy()
        self.current_indexes = [0, 1]
        self.width = 1
        self.right = 1
        self.left = 0
        self.end = 2
        self.index = 0
        self.current_obs = self.get_pair_data(self.current_indexes)
        self.test_cases_vector_temp = []
        return self.current_obs


    # Environment step to process agent´s action suggested order
    # Process one Mergesort step
    def step(self, test_case_index):
        # Analize action define test cases
        if test_case_index == 0:
            selected_test_case = self.test_cases_vector[self.current_indexes[0]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[1]]
        else:
            selected_test_case = self.test_cases_vector[self.current_indexes[1]]
            no_selected_test_case = self.test_cases_vector[self.current_indexes[0]]

        reward = self.calculate_reward(selected_test_case, no_selected_test_case)

        done = False
        ## an step of a merging sort
        if test_case_index == 1:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
            self.right = self.right + 1
            if self.right >= min(self.end, self.index + 2 * self.width):
                while self.left < self.index + self.width:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left + 1
        elif test_case_index == 0:
            self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
            self.left = self.left + 1
            if self.left >= self.index + self.width:
                while self.right < min(self.end, self.index + 2 * self.width):
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.right])
                    self.right = self.right + 1
        if self.right < self.end or self.left < self.index + self.width:
            None
        elif self.end < len(self.test_cases_vector):
            self.index = min(self.index + self.width * 2, len(self.test_cases_vector) - 1)
            self.left = self.index
            self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
            self.end = min(self.right + self.width, len(self.test_cases_vector))
            if self.right < self.left + self.width:
                while self.left < self.end:
                    self.test_cases_vector_temp.append(self.test_cases_vector[self.left])
                    self.left = self.left + 1
                self.width = self.width * 2
                self.test_cases_vector = self.test_cases_vector_temp.copy()
                self.test_cases_vector_temp = []
                self.index = 0
                self.left = self.index
                self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
                self.end = min(self.right + self.width, len(self.test_cases_vector))
        elif self.width < len(self.test_cases_vector) / 2:
            self.width = self.width * 2
            self.test_cases_vector = self.test_cases_vector_temp.copy()
            self.test_cases_vector_temp = []
            self.index = 0
            self.left = self.index
            self.right = min(self.left + self.width, len(self.test_cases_vector) - 1)
            self.end = min(self.right + self.width, len(self.test_cases_vector))
        else:
            done = True

            self.test_cases_vector = self.test_cases_vector_temp.copy()
            assert len(self.test_cases_vector) == len(self.cycle_logs.test_cases), "merge sort does not work as expected"
            self.sorted_test_cases_vector = self.test_cases_vector.copy()
            return self.current_obs, reward, done, {}

        if not done:
            self.current_indexes[0] = self.left
            self.current_indexes[1] = self.right
            self.current_obs = self._next_observation(test_case_index)

        return self.current_obs, reward, done, {}

    # next observation to send to the agent
    def _next_observation(self, index):
        # Next pair of test cases to order
        self.current_obs = self.get_pair_data(self.current_indexes)
        return self.current_obs
