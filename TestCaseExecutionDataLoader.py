# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from CICycleLog import CICycleLog

# Todo
#   Generalize the data loader to enriched datatsets and cycle reward data to implement better reward functions
#   for the environment. Make the data loader pardametrizable in data processing and data reward processing

# Loader of the data of the dataset
# Transforms dataset in CICycleLogs with useful test case information
class TestCaseExecutionDataLoader:
    def __init__(self, data_path, data_format):
        self.data_path = data_path
        self.data_format = data_format
        self.test_data = None
        self.min_cycle = 0
        self.max_cycle = 0
        self.cycle_count = 0

    # Load data from dataset
    def load_data(self):
        last_results = []
        cycle_ids = []
        max_size = 0

        print("Loading dataset ", self.data_path)
        df = pd.read_csv(self.data_path, error_bad_lines=False, sep=";")
        for i in range(df.shape[0]):
            last_result_str: str = df["LastResults"][i]
            temp_list = (last_result_str.strip("[").strip("]").split(","))
            if temp_list[0] != '':
                last_results.append(list(map(int, temp_list)))
            else:
                last_results.append([])
        df["LastResults"] = last_results
        self.test_data = df

        print("Done ")
        return self.test_data

    # Process
    def pre_process(self):
        print("Preprocessing dataset generating CI cycles to process ", self.data_path)
        self.min_cycle = min(self.test_data["Cycle"])
        self.max_cycle = max(self.test_data["Cycle"])
        ci_cycle_logs = []

        for i in range(self.min_cycle, self.max_cycle + 1):
            self.cycle_count += 1
            ci_cycle_log = CICycleLog(i)
            cycle_rew_data = self.test_data.loc[self.test_data['Cycle'] == i]

            for index, test_case in cycle_rew_data.iterrows():
                # ToDo This decision is maintained from original code:
                #   avg_exec_time=test_case["Duration"], last_exec_time=test_case["Duration"]
                #   Analyze alternatives
                ci_cycle_log.add_test_case(test_id=test_case["Id"], test_suite=test_case["Name"], avg_exec_time=test_case["Duration"],
                                           last_exec_time=test_case["Duration"], verdict=test_case["Verdict"], failure_history=test_case["LastResults"],
                                           cycle_id=test_case["Cycle"])

            ci_cycle_logs.append(ci_cycle_log)

        print("Done ")
        return ci_cycle_logs