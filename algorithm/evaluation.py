import numpy as np
import time
from algorithm.operator import Operator
import warnings
from joblib import Parallel, delayed
from algorithm.evaluator_accelerate import add_numba_decorator
import re
import concurrent.futures
from population.select import best_select


class Evaluation:
    def __init__(self, offspring_size, m, api_endpoint, api_key, llm_model, llm_use_local, llm_local_url, debug_mode,
                 problem, n_process, timeout, use_numba, lamda, alpha, **kwargs):

        # LLM settings
        self.offspring_size = offspring_size
        self.problem = problem
        self.operator = Operator(api_endpoint, api_key, llm_model, llm_use_local, llm_local_url, self.problem)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.n_process = n_process

        self.timeout = timeout
        self.use_numba = use_numba
        self.lamda = lamda
        self.alpha = alpha

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def replace_params(self, equ_code, equ_params):
        for i, param in enumerate(equ_params):
            equ_code = re.sub(rf'params\[{i}]', str(param), equ_code)
        equ_code = re.sub(r',\s*params', '', equ_code)
        return equ_code

    def init_population(self):
        n_create = 7
        population = []

        for i in range(n_create):
            _, pop = self.get_equation([], None, True)
            for p in pop:
                population.append(p)

        lamda_lst = []
        for ind in population:
            if ind['mse'] is not None and ind['S'] is not None:
                lamda = self.alpha * (ind['mse'] / ind['S'])
                lamda_lst.append(lamda)
        self.lamda = np.min(lamda_lst)
        for ind in population:
            if ind['mse'] is not None and ind['S'] is not None:
                ind['complex_reg'] = self.lamda * ind['S']
                ind['lamda'] = self.lamda
                ind['objective'] = ind['mse'] + self.lamda * ind['S']

        return population

    def init_population_seed(self, seeds):

        population = []

        fitness, opt_params_lst = Parallel(n_jobs=self.n_process)(
            delayed(self.problem.train_evaluate)(seed['code'])
            for seed in seeds)

        for i in range(len(seeds)):
            try:
                opt_params = np.round(np.array(opt_params_lst[i]), 5)
                equ_desc, equ_code = self.replace_params(seeds[i]['equation'], seeds[i]['code'], opt_params)
                seed_equ = {'equation': equ_desc, 'code': equ_code,
                            'opt_params': opt_params,
                            'objective': np.round(np.array(fitness[i]), 5), 'other_inf': None}

                population.append(seed_equ)

            except Exception as e:
                print("Error in seed equation")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def get_equation(self, pop, operator, init_flag=False):
        results = []
        try:
            results = []
            for _ in range(self.offspring_size):
                p, off = self.offspring_eval(pop, operator, init_flag)
                results.append((p, off))
        except Exception as e:
            if e:
                print(f"Error: {e}")
            print("Parallel time out .")

        time.sleep(2)

        out_p = []
        out_off = []

        for p, off in results:
            out_p.append(p)
            out_off.append(off)
        return out_p, out_off

    def offspring_eval(self, pop, operator, init_flag):
        try:
            p, offspring = self.get_offspring(pop, operator, init_flag)
            # print(offspring)
            if self.use_numba:
                pattern = r"def\s+(\w+)\s*\(.*\):"
                match = re.search(pattern, offspring['code'])

                function_name = match.group(1)

                code = add_numba_decorator(program=offspring['code'], function_name=function_name)
            else:
                code = offspring['code']

            n_retry = 1
            while self.check_duplicate(pop, offspring['code']):

                n_retry += 1

                p, offspring = self.get_offspring(pop, operator, init_flag)

                if self.use_numba:
                    pattern = r"def\s+(\w+)\s*\(.*\):"

                    # Search for function definitions in the code
                    match = re.search(pattern, offspring['code'])
                    function_name = match.group(1)
                    code = add_numba_decorator(program=offspring['code'], function_name=function_name)
                else:
                    code = offspring['code']

                if n_retry > 1:
                    break

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.problem.train_evaluate, code)
                mse, opt_params = future.result(timeout=self.timeout)
                offspring['mse'] = mse

                if mse is not None and opt_params is not None:
                    opt_params = np.round(opt_params, 5)
                    equ_code = self.replace_params(offspring['code'], opt_params)
                    offspring['param_code'] = equ_code
                    c = self.problem.cal_equ_complexity(offspring['param_code'])
                    offspring['complex'] = c
                    if c is not None:
                        c_reg = self.lamda * np.exp(c / 80)
                        fitness = mse + c_reg
                        offspring['S'] = np.exp(c / 80)
                        offspring['lamda'] = self.lamda
                        offspring['objective'] = fitness
                        offspring['complex_reg'] = c_reg
                    else:
                        offspring['objective'] = None

                else:
                    offspring['objective'] = None
                future.cancel()

        except Exception as e:
            print(e)
            offspring = {
                'knowledge': None,
                'insight': None,
                'code': None,
                'param_code': None,
                'mse': None,
                'complex': None,
                'S': None,
                'lamda': None,
                'complex_reg': None,
                'objective': None,
                'other_inf': None}
            p = None

        # Round the objective values
        return p, offspring

    def get_offspring(self, pop, operator, init_flag):
        offspring = {
            'knowledge': None,
            'insight': None,
            'code': None,
            'param_code': None,
            'mse': None,
            'complex': None,
            'S': None,
            'lamda': None,
            'complex_reg': None,
            'objective': None,
            'other_inf': None
        }
        if init_flag:
            parents = None
            [offspring['knowledge'], offspring['insight'], offspring['code']] = self.operator.init_equ()
        else:
            if operator:
                parents = best_select(pop, self.m)
                [offspring['knowledge'], offspring['insight'], offspring['code']] = self.operator.op_offspring(parents,
                                                                                                               operator)
            else:
                print(f"Evolution operator [{operator}] has not been implemented ! \n")

        return parents, offspring
