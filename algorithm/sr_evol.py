import numpy as np
import json
import time
import os
from scipy.stats import rankdata

from algorithm.evaluation import Evaluation
from population.management import population_management


class SrEvol:

    # initialization
    def __init__(self, paras, problem, **kwargs):

        self.problem = problem

        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        # self.use_local_llm = kwargs.get('use_local_llm', False)
        # assert isinstance(self.use_local_llm, bool)
        # if self.use_local_llm:
        #     assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
        #     assert isinstance(kwargs.get('url'), str)
        #     self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings
        self.pop_size = paras.pop_size
        self.offspring_size = paras.offspring_size
        self.max_fe = paras.max_fe

        self.operators_gen_num = paras.operators_gen_num
        if paras.m > self.pop_size or paras.m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.m = 2
        self.m = paras.m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.n_process = paras.n_process

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator
        self.lamda = paras.lamda
        self.alpha = paras.alpha

        print("- EvoSR-LLM parameters loaded -")

        # Set a random seed
        # random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                        print("duplicated result, retrying ... ")
            population.append(off)

    def cal_operator_fitness(self, operator_pop_fitness):
        merged_fitness = [value for operator_fitness in operator_pop_fitness for value in operator_fitness]
        ranks = rankdata(merged_fitness, method="average")

        average_ranks = []
        current_index = 0
        for operator_fitness in operator_pop_fitness:
            length = len(operator_fitness)
            operator_ranks = ranks[current_index:current_index + length]
            average_ranks.append(operator_ranks.mean())
            current_index += length
        return average_ranks

    def run(self):
        print("- Evolution Start -")
        time_start = time.time()
        gen = 0
        op_index = 0
        op_label_lst = ["MKP", "MIP", "MCP"]

        eval = Evaluation(self.offspring_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                          self.use_local_llm, self.llm_local_url,
                          self.debug_mode, self.problem, n_process=self.n_process,
                          timeout=self.timeout, use_numba=self.use_numba, lamda=self.lamda, alpha=self.alpha)
        operator_pop = eval.operator.operator_prompt_pop

        op_label = op_label_lst[op_index]
        new_op = eval.operator.add_new_op(op_label)
        print(f"-----Adding {op_label} OP!-----")
        operator_pop.append(new_op)
        operator_pop_fitness = [[] for _ in range(len(operator_pop))]
        op_index = (op_index + 1) % len(op_label_lst)
        # initialization
        population = []

        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = eval.init_population_seed(data)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=4)
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:
                print("creating initial population:")
                population = eval.init_population()
                for i in range(len(population)):
                    filename = (self.output_path
                                + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/samples/samples_fe={self.problem.fe - len(population) + i + 1}.json")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, 'w') as f:
                        if population[i]['code'] is not None:
                            json.dump(population[i], f, indent=4)

                population = population_management(population, self.pop_size)

                print(f"Pop initial: ")
                for i, off in enumerate(population):
                    if off['objective'] is not None:
                        print(" Obj: ", f"{off['objective']:.4e}", end="|")
                    else:
                        print(" Obj: ", str(off['objective']), end="|")

                print("initial population has been created!")

                # Save population to a file
                filename = (self.output_path
                            + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/pops/population_fe={self.problem.fe}.json")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=4)

        while self.problem.fe < self.max_fe:
            gen += 1
            for i in range(len(operator_pop)):
                op = operator_pop[i]
                print(f" OP: [{i + 1} / {len(operator_pop)}] ")
                parents, offsprings = eval.get_equation(population, op)

                self.add2pop(population, offsprings)
                for j, off in enumerate(offsprings):
                    filename = (self.output_path
                                + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/samples/samples_fe={self.problem.fe - len(offsprings) + j + 1}.json")
                    os.makedirs(os.path.dirname(filename), exist_ok=True)

                    if off['objective'] is not None:
                        if isinstance(off['objective'], np.complex128):
                            if off['objective'].imag == 0:
                                off['objective'] = off['objective'].real

                                print(" Obj: ", f"{off['objective']:.4e}", end="|")
                                operator_pop_fitness[i].append(off['objective'])

                                with open(filename, 'w') as f:
                                    json.dump(offsprings[j], f, indent=4)
                            else:
                                operator_pop_fitness[i].append(np.inf)

                        else:
                            print(" Obj: ", f"{off['objective']:.4e}", end="|")
                            operator_pop_fitness[i].append(off['objective'])

                            with open(filename, 'w') as f:
                                json.dump(offsprings[j], f, indent=4)

                    else:
                        print(" Obj: ", str(off['objective']), end="|")
                        operator_pop_fitness[i].append(np.inf)
                        with open(filename, 'w') as f:
                            json.dump(offsprings[j], f, indent=4)

                print(f"FEs used:{self.problem.fe}")

                size_act = min(len(population), self.pop_size)
                population = population_management(population, size_act)

            if gen == 20:
                self.lamda_change(population, eval)
                population = population_management(population, self.pop_size)

            if gen == self.operators_gen_num:
                gen = 0

                average_rank = self.cal_operator_fitness(operator_pop_fitness)
                filename = (self.output_path
                            + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/operator_pops/population_fe="
                            + str(self.problem.fe) + ".json")
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                operator_pop_data = {"operator_pop": operator_pop, "average_rank": average_rank}

                with open(filename, 'w') as f:
                    json.dump(operator_pop_data, f, indent=5)

                del_idx = average_rank.index(max(average_rank))
                operator_pop.pop(del_idx)

                op_label = op_label_lst[op_index]
                print(f"-----Adding {op_label} OP!-----")

                new_op = eval.operator.add_new_op(op_label)
                operator_pop.append(new_op)
                operator_pop_fitness = [[] for _ in range(len(operator_pop))]
                op_index = (op_index + 1) % len(op_label_lst)

            filename = (self.output_path
                        + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/pops/population_fe="
                        + str(self.problem.fe) + ".json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            vaild_nmse = self.problem.valid_evaluate(population[0]['param_code'])
            population[0].update({'test_id_nmse': vaild_nmse})
            test_nmse = self.problem.test_evaluate(population[0]['param_code'])
            population[0].update({'test_ood_nmse': test_nmse})

            filename = (self.output_path
                        + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/pops_best/population_fe="
                        + str(self.problem.fe) + ".json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            print(f"--- {self.problem.fe} of {self.max_fe} populations finished. "
                  f"Time Cost:  {((time.time() - time_start) / 60):.1f} m")
            print("Pop Objs: ", end=" ")
            for ind in population:
                if ind['objective'] is not None:
                    print(" Obj: ", f"{ind['objective']:.4e}", end="|")
                else:
                    print(" Obj: ", str(ind['objective']), end="|")
            print("")

        filename = (self.output_path
                    + f"/results/{self.problem.benchmark_name}/{self.llm_model}/{self.problem.problem_name}/best_equ.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        best_equ = population[0]
        best_equ["time_cost"] = time.time() - time_start
        with open(filename, 'w') as f:
            json.dump(best_equ, f, indent=5)

    def lamda_change(self, population, eval):
        lamda_lst = []
        for ind in population:
            if ind['mse'] is not None and ind['S'] is not None:
                lamda = self.alpha * (ind['mse'] / ind['S'])
                lamda_lst.append(lamda)
        eval.lamda = np.min(lamda_lst)
        for ind in population:
            if ind['mse'] is not None and ind['S'] is not None:
                ind['complex_reg'] = eval.lamda * ind['S']
                ind['lamda'] = eval.lamda
                ind['objective'] = ind['mse'] + eval.lamda * ind['S']
