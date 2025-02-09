import sys
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
sys.path.append(ROOT_PATH)
sys.path.append(ABS_PATH)

from algorithm.sr_evol import SrEvol
from utils.util import Paras
from Problems.problems import ProblemSR

paras = Paras()

paras.set_paras(benchmark="llm_sr",
                llm_api_endpoint=None,
                llm_api_key=None,
                llm_model="gpt-3.5-turbo",
                pop_size=10,
                offspring_size=2,
                max_fe=10000,
                n_process=4,
                operators_gen_num=120,
                exp_debug_mode=False,
                lamda=0.01,
                alpha=5)

problem_name = "bactgrow"
sr_problem = ProblemSR(paras.benchmark, problem_name)
evolution = SrEvol(paras, sr_problem)

evolution.run()


