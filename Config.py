# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021

import pickle
import json

# Configuration file
# Contains framework, algo , algo parameters, processing configurations
# Allows to execute with different configurations

# ToDo
#  Now all parameters are saved and loaded independently if they are used.
#  Extend to other parameters of the supported algorithms
class Config:
    def __init__(self):
        # Default settings
        self.library = "tensorflow"
        self.mode = "pairwise"
        self.algo = "a2c"
        self.dataset_type = "simple"
        self.padding_digit = -1
        self.win_size = -1

        self.episodes = 1000
        self.steps = 1000
        self.training_steps = 1000

        self.max_list_size = 0
        self.min_fails = 1
        self.min_cases_per_episode = 6
        self.max_test_cases_count = 400

        self.start_cycle = 1
        self.end_cycle = 0
        self.cycle_count = 0
        self.train_data = "./data/bt.csv"
        self.output_path = "./data/A2C"
        self.log_file = "btlog.csv"

        self.verbose = 0
        self.tensorboard_log = None

        self.discount_factor = 0.9
        self.experience_replay = True
        self.gamma = 0.90
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.replay_ratio = 0
        self.buffer_size = 10000
        self.exploration_fraction = 1
        self.exploration_final_eps = 0.02
        self.exploration_initial_eps = 1.0
        self.train_freq = 1
        self.double_q = True
        self.learning_starts = 1000
        self.target_network_update_freq = 500
        self.prioritized_replay = False
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4

    # Save configuration file to disk
    def save(self, file):
        #with open(file, 'wb') as f:
        #    pickle.dump(self, f)
        jsonStr = json.dumps(self.__dict__)
        jsonFile = open(file, "w")
        jsonFile.write(jsonStr)
        jsonFile.close()

    # Live configuration file from disk
    def load(self,file):
        #with open(file) as f:
        #    loaded_obj = pickle.load(f)
        #    return loaded_obj
        fileObject = open(file, "r")
        jsonContent = fileObject.read()
        self.__dict__ = json.loads(jsonContent)

