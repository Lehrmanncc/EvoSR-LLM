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
                llm_api_endpoint="aihubmix.com",
                llm_api_key="sk-JvMrrSGpFqx1I9q6Fa5bFe8a8f5f45F0A7A5EbE220F032Bd",
                llm_model="gpt-4o-mini",
                pop_size=10,
                offspring_size=2,
                max_fe=3000,
                n_process=4,
                operators_gen_num=120,
                alpha=0.01,
                exp_output_path = "./results_gpt4o_mini_25_9_9")

problem_name = "bactgrow"
sr_problem = ProblemSR(paras.benchmark, problem_name, None)
evolution = SrEvol(paras, sr_problem)

evolution.run()


