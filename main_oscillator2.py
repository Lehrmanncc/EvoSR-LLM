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
idx = 1

# Set parameters #
paras.set_paras(benchmark="llm_srbench",
                llm_api_endpoint="aihubmix.com",
                # llm_api_key=os.environ["XIDIAN_API_KEY"],  # set your key
                llm_api_key="sk-cG0D0wA0Vo1ESfHIFaFbF90bEd514d9e8cC7175b26AdEb52",
                # llm_model="gpt-3.5-turbo",  # set llm
                llm_model="gpt-4o-mini",
                pop_size=10,
                offspring_size=2,
                max_fe=3000,
                operators_gen_num=120,
                n_process=4,
                exp_debug_mode=False,
                lamda=0.00001,
                alpha=5,
                exp_output_path=f"./results_25_8_10/ins{idx}")


# problem_path = os.path.join(os.getcwd(), "Problems", paras.benchmark)
# equ_dataset_dir = [entry.name for entry in os.scandir(problem_path) if entry.is_dir()]
# for problem_name in equ_dataset_dir:
# problem_name = "oscillator2"
problem_name = "chem"
sr_problem = ProblemSR(paras.benchmark, problem_name, idx)
evolution = SrEvol(paras, sr_problem)

evolution.run()


