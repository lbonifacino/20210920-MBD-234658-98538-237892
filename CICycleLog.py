# Based on the sources from https://github.com/moji1/tp_rl  from paper https://arxiv.org/pdf/2011.01834.pdf
# Reinforcement Learning for Test Case Prioritization
# Mojtaba Bagherzadeh, Nafiseh Kahani, and Lionel Briand, Fellow, IEEE

# Some sources are taken as if from the original project
# Other has been adapted to the needs of this project
# mbegerez 2021


import numpy as np
import random
import copy
from sklearn import preprocessing

# Integration Cycle, id with test cases
class CICycleLog:
    '''List of test cases'''
    test_cases = {}
    '''Cycle Id'''
    cycle_id = 0

    def __init__(self, cycle_id: int):
        self.cycle_id = cycle_id
        self.test_cases = []

    # Add test case
    def add_test_case(self,  cycle_id, test_id, test_suite, avg_exec_time: int, last_exec_time: int, verdict: int, failure_history: list):
        #Generate new test case
        test_case:dict = {}
        test_case['test_id'] = test_id
        test_case['test_suite'] = test_suite

        test_case['avg_exec_time'] = avg_exec_time
        test_case['verdict'] = verdict
        test_case['last_exec_time'] = last_exec_time
        test_case['cycle_id'] = cycle_id
        test_case['last_exec_time'] = last_exec_time

        if failure_history:
            test_case['failure_history'] = failure_history
            test_case['age'] = len(failure_history)
        else:
            test_case['failure_history'] = []
            test_case['age'] = 0

        self.test_cases.append(test_case)

    def get_test_case_vector_length(self,test_case,win_size):
        extra_length = 4
        return win_size + extra_length

    def get_test_cases_count(self) -> object:
        return len(self.test_cases)

    def get_failed_test_cases_count(self):
        cnt = 0
        for test_case in self.test_cases:
            if test_case['verdict'] == 1:
                cnt = cnt + 1
        return cnt

    def export_test_cases(self, option: str, pad_digit=9, max_test_cases_count=0, winsize=4, test_case_vector_size=7):
        if option == "list_avg_exec_with_failed_history":
            # assume param1 refers to the number of test cases,
            # params 2 refers to the history windows size, and param3 refers to pa
            test_cases_array = np.zeros((max_test_cases_count, test_case_vector_size))

            i = 0
            for test_case in self.test_cases:
                test_cases_array[i] = self.export_test_case(test_case, "list_avg_exec_with_failed_history", win_size=winsize)
                i = i + 1

            for i in range(len(self.test_cases), max_test_cases_count):
                test_cases_array[i] = np.repeat(pad_digit, test_case_vector_size)

            test_cases_array = preprocessing.normalize(test_cases_array, axis=0, norm='max')

            return test_cases_array
        else:
            return None

    def export_test_case(self, test_case: dict, option: str, pad_digit=9, win_size=4):
        extra_length = 4

        test_case_vector = np.zeros((win_size + extra_length))

        index_1 = 0
        for j in range(0, len(test_case['failure_history'])):
            if j >= win_size:
                break
            test_case_vector[j] = test_case['failure_history'][j]
            index_1 = index_1 +1

        for j in range(len(test_case['failure_history']), win_size):
            test_case_vector[j] = pad_digit
            index_1 = index_1 +1

        test_case_vector[index_1] = test_case['avg_exec_time']
        test_case_vector[index_1+1] = test_case['age']

        if 'time_group' in test_case.keys():
            test_case_vector[index_1 + 2] = test_case['time_group']
        else:
            test_case_vector[index_1 + 2] = 0

        if 'duration_group' in test_case.keys():
            test_case_vector[index_1 + 3] = test_case['duration_group']
        else:
            test_case_vector[index_1 + 3] = 0

        return test_case_vector


    def calc_APFD_vector_prob(self, test_case_vector_prob: list, threshold: float):
        sum_ranks: float = 0
        apfd: float = 0

        i = 1
        test_case_vector_prob = sorted(test_case_vector_prob, key=lambda x: x['prob'])
        for test_case_prob in test_case_vector_prob:
            sum_ranks = sum_ranks + self.test_cases[test_case_prob['index']]['verdict'] * i
            i = i+1

        N: float = self.get_test_cases_count()
        M: float = self.get_failed_test_cases_count()

        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))

        return apfd

    def calc_APFD_ordered_vector(self, test_case_vector: list):
        sum_ranks: float = 0
        apfd: float = 0

        i = 1
        for test_case in test_case_vector:
            sum_ranks = sum_ranks + test_case['verdict'] * i
            i = i+1

        N:float = self.get_test_cases_count()
        M:float = self.get_failed_test_cases_count()

        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))

        return apfd

    def calc_RPA_vector(self, test_case_vector: list):
        ranks = []
        optimal_order = self.get_optimal_order()

        i = 0
        for test_case in test_case_vector:
            ranks.append(self.get_test_cases_count() - optimal_order.index(test_case))

        return self.calc_score_ranking(ranks)

    #Optimal RPA
    def get_optimal_RPA(self,n:int):
        if n == 1:
            return 1
        else:
            return (n*n) + self.get_optimal_RPA(n-1)

    # Ratio RPA/Optimal RPA
    def calc_NRPA_vector(self,test_case_vector: list):
        RPA = self.calc_RPA_vector(test_case_vector)
        ORPA = self.get_optimal_RPA(self.get_test_cases_count())

        return RPA/ORPA

    def calc_score_ranking(self, ranks: list):
        if not ranks:
            return 0
        elif len(ranks) <= 1:
            return ranks[0]
        else:
            return ranks[0]*len(ranks) + self.calc_score_ranking(ranks[1:])

    # APFD of ordered test cases
    def calc_APFD(self, ordered_test_cases_id):
        sum_ranks: float = 0
        apfd: float = 0
        ordered_test_cases_temp = []

        for test_case_id in ordered_test_cases_id:
            if test_case_id < self.get_test_cases_count():
                ordered_test_cases_temp.append(test_case_id)

        ordered_test_cases = ordered_test_cases_temp
        for i in range(0, len(ordered_test_cases)):
            if ordered_test_cases[i] < self.get_test_cases_count():
                sum_ranks = sum_ranks + self.test_cases[ordered_test_cases[i]]['verdict'] * (i + 1)

        N:float = self.get_test_cases_count()
        M:float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))

        return apfd

    # APFD Random Order
    def calc_random_APFD(self):
        random_order = []
        while len(random_order) < self.get_test_cases_count():
            rand_num = random.randint(0, self.get_test_cases_count())
            if random_order.count(rand_num) <= 0:
                random_order.append(rand_num)

        random_apfd = self.calc_APFD(random_order)

        return random_apfd

    # APFD Optimal Order
    def calc_optimal_APFD(self):
        # Optimal order
        optimal_order = sorted(self.test_cases, key=lambda x: x['verdict'], reverse=True)

        sum_ranks = 0
        i = 1
        apfd: float = 0
        for test_case in optimal_order:
            sum_ranks = sum_ranks + test_case['verdict'] * i
            i = i+1

        N: float = self.get_test_cases_count()
        M: float = self.get_failed_test_cases_count()
        if N > 0 and M > 0:
            apfd = 1 - (sum_ranks / (N * M)) + (1 / (2 * N))

        return apfd

    # Order by optimal order
    # Order by failed test and execution time
    # Failed test cases minimun time first
    def get_optimal_order(self):
        optimal_order_by_verdict = copy.deepcopy(sorted(self.test_cases, key=lambda x: x['verdict'], reverse=True))
        optimal_order = []
        optimal_order.extend(sorted(optimal_order_by_verdict[0:self.get_failed_test_cases_count()], key=lambda x: x['last_exec_time']))
        optimal_order.extend(sorted(optimal_order_by_verdict[self.get_failed_test_cases_count():], key=lambda x: x['last_exec_time']))
        return optimal_order


