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
idx = 1

paras.set_paras(benchmark="llm_srbench",
                llm_api_endpoint="aihubmix.com",
                llm_api_key="sk-xHQx5tvpC6Q30Qrk2c5e57665613441286E057C181CbB062",
                llm_model="gpt-4o-mini",
                # llm_model="gpt-3.5-turbo",
                pop_size=10,
                offspring_size=2,
                max_fe=3000,
                n_process=4,
                operators_gen_num=120,
                alpha=5,
                exp_output_path=f"./results_25_8_10/ins{idx}")

# problem_name = "bactgrow"
problem_name = "matsci"

sr_problem = ProblemSR(paras.benchmark, problem_name, idx)
evolution = SrEvol(paras, sr_problem)

evolution.run()


