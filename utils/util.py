import os
import shutil
import warnings
import types
import sys
import re


class Paras:
    def __init__(self):

        self.problem = None
        self.benchmark = None
        self.selection = None
        self.management = None

        self.pop_size = 5
        self.offspring_size = 2
        self.max_fe = 200
        self.operators = None
        self.m = 2

        self.llm_use_local = False  # if use local model
        self.llm_local_url = None  # your local server 'http://127.0.0.1:11012/completions'
        self.llm_api_endpoint = None  # endpoint for remote LLM, e.g., api.deepseek.com
        self.llm_api_key = None  # API key for remote LLM, e.g., sk-xxxx
        self.llm_model = None  # model type for remote LLM, e.g., deepseek-chat

        self.exp_debug_mode = False  # if debug
        self.exp_output_path = "./"  # default folder for ael outputs
        self.exp_use_seed = False
        self.exp_seed_path = "./seeds/seeds.json"
        self.exp_use_continue = False
        self.exp_continue_id = 0
        self.exp_continue_path = "./results/pops/population_generation_0.json"
        self.n_process = 4


        self.eva_timeout = 30
        self.eva_numba_decorator = False
        self.operators_gen_num = 20
        self.lamda = 0.001
        self.alpha = 0.1

    def set_parallel(self):
        import multiprocessing
        num_processes = multiprocessing.cpu_count()
        if self.n_process == -1 or self.n_process > num_processes:
            self.exp_n_proc = num_processes
            print(f"Set the number of proc to {num_processes} .")

    def set_paras(self, *args, **kwargs):

        # Map paras
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Identify and set parallel
        self.set_parallel()


def create_folders(results_path):
    # Specify the path where you want to create the folder
    folder_path = os.path.join(results_path, "results")

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # Remove the existing folder and its contents
        #shutil.rmtree(folder_path)

        # Create the main folder "results"
        os.makedirs(folder_path)

    # Create subfolders inside "results"
    subfolders = ["history", "pops", "pops_best"]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)


def code_str2code(code_str):
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create a new module object
        sr_module = types.ModuleType("sr_module")

        # Execute the code string in the new module's namespace
        exec(code_str, sr_module.__dict__)

        # Add the module to sys.modules so it can be imported
        sys.modules[sr_module.__name__] = sr_module

        return sr_module


def mk_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        pass


def get_var_num(code_str):
    match = re.search(r'def\s+\w+\((.*?)\):', code_str)
    if match:
        arguments = [arg.strip() for arg in match.group(1).split(',') if arg.strip() != 'params']
        return len(arguments)
    else:
        return None


def get_params_num(code_str):
    indices = re.findall(r'params\[(\d+)\]', code_str)
    if indices:
        return len(indices)
    else:
        return None


