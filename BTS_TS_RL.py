# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021

# This is the principal program
#   Train an agent for prioritization of test cases in a CI cycle
#   Predict prioritization of test cases in a CI cycle
#
# In the original source the predict and train occurred against an external dataset. Learning is offline.
# There was no interaction with the CI environment, so the code simulated it.
# The dataset used had the test cases to prioritize and the true values of the execution of
#
# In our solution we have the CI and test execution environments available, so we can first predict and then execute
# the prediction over the test execution environment, then train the agent with the execution results.

import os
import math
import argparse
from pathlib import Path
from datetime import datetime
from statistics import mean
from Config import Config
from TestCaseExecutionDataLoader import TestCaseExecutionDataLoader
from CustomCallback import CustomCallback
from TPPairWiseAgent import TPPairWiseAgent
from TPPairWiseAgent2 import TPPairWiseAgent2
from CIPairWiseEnv import CIPairWiseEnv
from CIPairWiseEnvACER import CIPairWiseEnvACER
from Monitor import Monitor

# Reward function
# todo
#   Generalize reward functions and associated enriched dataset.
#   Make datasets and environments more general.

# Simple reward function to pass to the environment.
def calculate_reward(selected_test_case, no_selected_test_case):
    if selected_test_case['verdict'] > no_selected_test_case['verdict']:
        reward = 1
    elif selected_test_case['verdict'] < no_selected_test_case['verdict']:
        reward = 0
    elif selected_test_case['last_exec_time'] <= no_selected_test_case['last_exec_time']:
        reward = .5
    elif selected_test_case['last_exec_time'] > no_selected_test_case['last_exec_time']:
        reward = 0
    return reward

# Train the agent
def train(conf):

    # Parameters
    mode = conf.mode
    algo = conf.algo.upper()
    episodes = conf.episodes
    start_cycle = conf.start_cycle
    end_cycle = conf.end_cycle
    cycle_count = conf.cycle_count
    verbose = conf.verbose
    model_path = conf.output_path

    test_case_data = ci_cycle_logs
    dataset_name = ""

    # Log directories
    log_dir = os.path.dirname(conf.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log file
    log_file = open(conf.log_file, "a")
    conf.log_file = log_file

    # Metrics
    # The metrics corresponds to the input dataset, this dataset contains all test cases from the CI ordered in the prediction order
    # and veredict and time execution from the last test case executions.
    log_file_test_cases = open(conf.train_data.replace(".csv","_Metrics.csv"), "w")
    # Metric file fields
    log_file_test_cases.write("Cycle;Ids;Names;APFD;NRPA;Optimal APFD;Random APFD;Failed;Total\n")
    conf.log_file_test_cases = log_file_test_cases

    # Log file fields
    log_file.write(
        "timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd" + os.linesep)


    first_round: bool = True

    conf.calculate_reward = calculate_reward
    print("Experiment from ", start_cycle, " to ", end_cycle)
    tp_agent = None
    model_save_path = model_path + "current_" + mode + "_" + algo + dataset_name
    apfds = []
    nrpas = []

    # Decide what framework to use (Tensorflow stable-baselines implementations, Pytorch statble-baselines3)
    if (conf.library == "pytorch"):
        print(" Pytorch ")
        model_save_path = model_save_path + "_p"
    else:
        print(" Tensorflow ")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        model_save_path = model_save_path + "_t"

    # File to save agent
    model_save_file = model_save_path + ".zip"

    # For each CI cycle
    for i in range(cycle_count):
        # Cycle test cases
        test_case_vector = test_case_data[i].test_cases.copy()
        # Cycle ID
        Ci = test_case_data[i].cycle_id
        # Cyle number of tests
        N = test_case_data[i].get_test_cases_count()
        # Cycle number of failed test cases
        F = test_case_data[i].get_failed_test_cases_count()
        # Decide if this cycle is processed
        if (N < conf.min_cases_per_episode) or \
                ((conf.dataset_type == "simple") and
                 (F < conf.min_fails)):
            print("Cycle ", Ci, " with ", N, " test cases , ", F ," failed.")
            continue

        print("Cycle ", Ci, " with ", N, " test cases , ", F ," failed.")
        # Steps need to apply mergesort to order all test cases in  the Cycle
        steps = int(episodes * (N * (math.log(N, 2) + 1)))
        # ACER has a different Observation space for the environment
        if algo.upper() == "ACER":
            env = CIPairWiseEnvACER(test_case_data[i], conf)
        else:
            env = CIPairWiseEnv(test_case_data[i], conf)

        # Generate monitor
        env = Monitor(env, model_save_path + "_monitor.csv")
        callback_class = CustomCallback(save_path=model_save_path,
                                        check_freq=int(conf.steps / conf.episodes), log_dir=log_dir,
                                        verbose=conf.verbose)

        # If first cycle
        if first_round:
            # If the agent has been previously trained
            if os.path.isfile(model_save_file):
                print(" Load Agent from ", model_save_file)
                #Load agent from disk, according to implementation used
                if(conf.library == "pytorch"):
                    tp_agent = TPPairWiseAgent.load_model(env=env, algo = algo, path=model_save_file)
                else:
                    tp_agent = TPPairWiseAgent2.load_model(env=env, algo = algo, path=model_save_file)
            # Else create agent
            else:
                print(" Create Agent ")
                # Create agent, according to implementation used
                if (conf.library == "pytorch"):
                    tp_agent = TPPairWiseAgent.create_model(env, conf)
                else:
                    tp_agent = TPPairWiseAgent2.create_model(env, conf)

            training_start_time = datetime.now()
            print(" Start learn model")
            # Train the agent, according to implementation used
            if (conf.library == "pytorch"):
                TPPairWiseAgent.train_agent(env=env, conf=conf, path_to_save_agent=model_save_path, base_model=tp_agent, callback_class=callback_class)
            else:
                TPPairWiseAgent2.train_agent(env=env, conf=conf, path_to_save_agent=model_save_path, base_model=tp_agent, callback_class=callback_class)
            training_end_time = datetime.now()
            print(" End learn model ", training_end_time)
            first_round = False
        else:
            print(" Load previous model from ", model_save_file)
            # Load agent from disk, according to implementation used
            if (conf.library == "pytorch"):
                tp_agent = TPPairWiseAgent.load_model(env=env, algo=algo,  path=model_save_file)
            else:
                tp_agent = TPPairWiseAgent2.load_model(env=env, algo=algo, path=model_save_file)
            training_start_time = datetime.now()
            print(" Start learn model ", training_start_time)
            # Train agent, according to implementation used
            if (conf.library == "pytorch"):
                TPPairWiseAgent.train_agent(env=env, conf=conf, path_to_save_agent=model_save_path, base_model=tp_agent, callback_class=callback_class)
            else:
                TPPairWiseAgent2.train_agent(env=env, conf=conf, path_to_save_agent=model_save_path, base_model=tp_agent, callback_class=callback_class)
            training_end_time = datetime.now()
            print(" End learn model ", training_end_time)

        # Analize metrics
        test_time_end = datetime.now()
        test_case_id_vector = []
        test_case_name_vector = []
        #test_case_vector = test_case_data[i].test_cases.copy()
        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            test_case_name_vector.append(str(test_case['test_suite']))
            cycle_id_text = str(test_case['cycle_id'])

        # APFD
        if F != 0:
            apfd = test_case_data[i].calc_APFD_ordered_vector(test_case_vector)
            apfd_optimal = test_case_data[i].calc_optimal_APFD()
            apfd_random = test_case_data[i].calc_random_APFD()
            apfds.append(apfd)
        else:
            apfd =0
            apfd_optimal =0
            apfd_random =0

        # NRPA
        nrpa = test_case_data[i].calc_NRPA_vector(test_case_vector)
        nrpas.append(nrpa)
        # Show metrics
        print("Training agent on cycle " + str(Ci) +
              " resulted in APFD: " + str(apfd) +
              " , NRPA: " + str(nrpa) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(F) +
              " , # test cases: " + str(N), flush=True)

        # write log
        log_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(0) +
                       "," + str(0) + "," + str(conf.win_size) + "," +
                       str(N) + "," +
                       str(F) + "," + str(apfd) + "," +
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
        #log_file_test_cases.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
        #               + Path(model_save_path).stem + "," +
        #               str(episodes) + "," + str(steps) + "," + str(cycle_iexced_text) + "," + str(0) +
        #               "," + str(0) + "," + str(conf.win_size) + "," +
        #                          (';'.join(test_case_id_vector)) + os.linesep)

        # Show average metrics
        if (len(apfds)):
            print(f"    Average apfd so far is {mean(apfds)}")
        print(f"    Average nrpas so far is {mean(nrpas)}")

        # Metrics file info
        log_file_test_cases.write(cycle_id_text + ";[" + (','.join(test_case_id_vector)) + "];[" + (','.join(test_case_name_vector)) + "];" +  str(apfd) + ";" + str(nrpa) + ";" + str(apfd_optimal) + ";" + str(apfd_random) + ";" +  str(F) + ";" + str(N)  +  "\n")

        log_file.flush()
        log_file_test_cases.flush()

    log_file.close()
    log_file_test_cases.close()

# Predict with the Agent
def predict(conf):
    # Parameters
    mode = conf.mode
    algo = conf.algo.upper()
    episodes = conf.episodes
    start_cycle = conf.start_cycle
    end_cycle = conf.end_cycle
    cycle_count = conf.cycle_count
    verbose = conf.verbose
    model_path = conf.output_path

    test_case_data = ci_cycle_logs
    dataset_name = ""

    log_dir = os.path.dirname(conf.log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log file
    log_file = open(conf.log_file, "a")
    conf.log_file = log_file

    log_file_test_cases = open("./data/BTS_TS_Prediction_" + str(start_cycle) + ".csv", "w")
    log_file_test_cases.write("Cycle;Ids;Names;APFD;NRPA;Optimal APFD;Random APFD;Failed;Total\n")
    conf.log_file_test_cases = log_file_test_cases

    log_file.write(
        "timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd" + os.linesep)

    first_round: bool = True

    model_save_path = None
    apfds = []
    nrpas = []

    conf.calculate_reward = calculate_reward

    print("Experiment from ", start_cycle, " to ", end_cycle)
    tp_agent = None
    model_save_path = model_path + "current_" + mode + "_" + algo + dataset_name

    # If exists load agent
    if (conf.library == "pytorch"):
        print(" Pytorch ")
        model_save_path = model_save_path + "_p.zip"
    else:
        print(" Tensorflow ")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        model_save_path = model_save_path + "_t.zip"
    model_save_file = model_save_path + ".zip"

    for i in range(cycle_count):
        test_case_vector = []

        Ci = test_case_data[i].cycle_id
        N = test_case_data[i].get_test_cases_count()
        F = test_case_data[i].get_failed_test_cases_count()
        if (N < conf.min_cases_per_episode) or \
                ((conf.dataset_type == "simple") and
                 (F < conf.min_fails)):
            print("Cycle ", Ci, " with ", N, " test cases , ", F ," failed.")
            continue

        print("Cycle ", Ci, " with ", N, " test cases , ", F ," failed.")
        steps = int(episodes * (N * (math.log(N, 2) + 1)))
        if algo.upper() == "ACER":
            env = CIPairWiseEnvACER(test_case_data[i], conf)
        else:
            env = CIPairWiseEnv(test_case_data[i], conf)
        #Generate Monitor
        env = Monitor(env, model_save_path + "_monitor.csv")
        callback_class = CustomCallback(save_path=model_save_path,
                                        check_freq=int(conf.steps / conf.episodes), log_dir=log_dir,
                                        verbose=conf.verbose)

        if first_round:
            # If exists load agent
            print(" Load Agent from " + model_save_file)
            if os.path.isfile(model_save_file):
                if (conf.library == "pytorch"):
                    tp_agent = TPPairWiseAgent.load_model(env=env, algo=algo, path=model_save_file)
                else:
                    tp_agent = TPPairWiseAgent2.load_model(env=env, algo=algo, path=model_save_file)
            # Else create agent
            else:
                return

            training_start_time = datetime.now()
            print(" Start test model ", training_start_time)
            if (conf.library == "pytorch"):
                test_case_vector = TPPairWiseAgent.test_agent(env=env, model_path=model_save_path, model=tp_agent)
            else:
                test_case_vector = TPPairWiseAgent2.test_agent(env=env, model_path=model_save_path, model=tp_agent)
            training_end_time = datetime.now()
            print(" End test model ", training_end_time)
            first_round = False
        else:
            print(" Load previous model from ", model_save_file)
            if (conf.library == "pytorch"):
                tp_agent = TPPairWiseAgent.load_model(env=env, algo=algo, path=model_save_file)
            else:
                tp_agent = TPPairWiseAgent2.load_model(env=env, algo=algo, path=model_save_file)
            training_start_time = datetime.now()
            print(" Start test model ", training_start_time)
            if (conf.library == "pytorch"):
                test_case_vector = TPPairWiseAgent.test_agent(env=env, model_path=model_save_path, model=tp_agent)
            else:
                test_case_vector = TPPairWiseAgent2.test_agent(env=env, model_path=model_save_path, model=tp_agent)
            training_end_time = datetime.now()
            print(" End test model ", training_end_time)

        test_time_end = datetime.now()
        test_case_id_vector = []
        test_case_name_vector = []
        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            test_case_name_vector.append(str(test_case['test_suite']))
            cycle_id_text = str(test_case['cycle_id'])

        if F != 0:
            apfd = test_case_data[i].calc_APFD_ordered_vector(test_case_vector)
            apfd_optimal = test_case_data[i].calc_optimal_APFD()
            apfd_random = test_case_data[i].calc_random_APFD()
            apfds.append(apfd)
        else:
            apfd =0
            apfd_optimal =0
            apfd_random =0
        nrpa = test_case_data[i].calc_NRPA_vector(test_case_vector)
        nrpas.append(nrpa)

        print("Testing agent on cycle " + str(Ci) +
              " resulted in APFD: " + str(apfd) +
              " , NRPA: " + str(nrpa) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(F) +
              " , # test cases: " + str(N), flush=True)
        log_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(0) +
                       "," + str(0) + "," + str(conf.win_size) + "," +
                       str(N) + "," +
                       str(F) + "," + str(apfd) + "," +
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) + os.linesep)
        #log_file_test_cases.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
        #               + Path(model_save_path).stem + "," +
        #               str(episodes) + "," + str(steps) + "," + str(cycle_iexced_text) + "," + str(0) +
        #               "," + str(0) + "," + str(conf.win_size) + "," +
        #                          (';'.join(test_case_id_vector)) + os.linesep)
        if (len(apfds)):
            print(f"    Average apfd so far is {mean(apfds)}")
        print(f"    Average nrpas so far is {mean(nrpas)}")
        log_file_test_cases.write(cycle_id_text + ";[" + (','.join(test_case_id_vector)) + "];[" + (','.join(test_case_name_vector)) + "];" +  str(apfd) + ";" + str(nrpa) + ";" + str(apfd_optimal) + ";" + str(apfd_random) + ";" +  str(F) + ";" + str(N)  +  "\n")

        log_file.flush()
        log_file_test_cases.flush()

    log_file.close()
    log_file_test_cases.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BTS TS RL')

    parser.add_argument('-f', '--file', help='Input file', required=True)
    parser.add_argument('-a', '--action', help='[train,predict]', required=True)
    parser.add_argument('-m', '--algorithm', help='Pytorch [a2c,ppo,dqn] Tensorflow [dqn,ppo2,td3,a2c,acer,acktr,ppo1,trpo,ddpg,sac]', required=False)
    parser.add_argument('-c', '--configfile', help='config json file ', required=False)

    args = parser.parse_args()

    if not args.algorithm:
        algo = 'a2c'
    else:
        algo = args.algoritm

    conf = Config()
    conf.dataset_type = "simple"
    conf.win_size = 10
    conf.start_cycle = 1
    conf.end_cycle = 1
    conf.cycle_count = 1
    conf.mode = 'pairwise'
    conf.algo = algo
    conf.episodes = 1000
    conf.steps = 1000

    conf.min_fails = 0

    if not args.configfile:
        config = "./data/Config.json"
    else:
        config = args.configfile

    if os.path.exists(config):
        conf.load(config)
        print("Loading configuration from ", config)
    else:
        print("Loading default configuration")

    conf.train_data = args.file
    conf.dataset_name = Path(conf.train_data).stem
    conf.output_path = './bts_ts_rl/' + conf.mode + "/" + conf.library + "/" + conf.algo + "/bt_" + str(conf.win_size) + "/"
    conf.log_file = conf.output_path + conf.mode + "_" + conf.algo + "_" + conf.dataset_name + "_" + str(
        conf.episodes) + "_" + str(conf.win_size) + "_log.txt"

    # Load data
    test_data_loader = TestCaseExecutionDataLoader(conf.train_data, conf.dataset_type)
    test_data = test_data_loader.load_data()
    ci_cycle_logs = test_data_loader.pre_process()

    conf.cycle_count = test_data_loader.cycle_count
    conf.start_cycle = test_data_loader.min_cycle
    conf.end_cycle = test_data_loader.max_cycle
    conf.save(config)

    action = args.action
    if action == "train":
        train(conf=conf)
    if action == "predict":
        predict(conf=conf)


