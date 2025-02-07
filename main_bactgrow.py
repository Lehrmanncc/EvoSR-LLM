### Test Only ###
# Set system path
import sys
import os

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(ABS_PATH, "..", "..")
sys.path.append(ROOT_PATH)  # This is for finding all the modules
sys.path.append(ABS_PATH)
print(ABS_PATH)
from algorithm.sr_evol import SrEvol
from utils.util import Paras
from Problems.problems import ProblemSR

# Parameter initilization #
paras = Paras()

# Set parameters #
paras.set_paras(benchmark="llm_sr",
                llm_api_endpoint="api.vveai.com",
                # llm_api_key=os.environ["XIDIAN_API_KEY"],  # set your key
                llm_api_key="sk-IAKuTTcWHDLIrkWQCe9f6e43Af7d445bBaC262D876980836",
                llm_model="gpt-3.5-turbo",  # set llm
                # llm_model="gpt-4o-mini",
                pop_size=10,
                offspring_size=2,
                max_fe=10000,
                n_process=4,
                operators_gen_num=120,
                exp_debug_mode=False,
                lamda=0.01,
                alpha=5)


# problem_path = os.path.join(os.getcwd(), "Problems", paras.benchmark)
# equ_dataset_dir = [entry.name for entry in os.scandir(problem_path) if entry.is_dir()]
# for problem_name in equ_dataset_dir:
problem_name = "bactgrow"
sr_problem = ProblemSR(paras.benchmark, problem_name)
evolution = SrEvol(paras, sr_problem)

evolution.run()


