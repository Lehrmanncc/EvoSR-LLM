import numpy as np
import os
import json
import types
import sys
import warnings
import re
from utils.util import mk_dir


def get_params_num(code_str):
    # 匹配所有 params[数字] 形式的字符串
    indices = re.findall(r'params\[(\d+)\]', code_str)
    if indices:
        return len(indices)
    else:
        return None


def test_evaluate(test_id_data, test_ood_data, sr_code_str):
    sr_module = code_str2code(sr_code_str, "sr1")
    if sr_module is not None:
        test_id_nmse = test_eval(test_id_data, sr_module)
        test_ood_nmse = test_eval(test_ood_data, sr_module)

        return test_id_nmse, test_ood_nmse
    else:
        return None, None


def test_eval(test_data, sr_code):
    try:
        y_pred = sr_code.equation(*test_data.T[:-1])
        nmse = np.mean((test_data[:, -1] - y_pred) ** 2) / np.var(test_data[:, -1])
        return nmse
    except Exception as e:
        print("evaluate an error occurred:", e)
        return None


def code_str2code(code_str, name):
    # Suppress warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Create a new module object
            sr_module = types.ModuleType(name)

            # Execute the code string in the new module's namespace
            exec(code_str, sr_module.__dict__)
            # exec(code_str, {})

            # Add the module to sys.modules so it can be imported
            sys.modules[sr_module.__name__] = sr_module

            return sr_module

    except Exception as e:
        print(f"Execution Error: {e}")
        return None


# def test_evaluate(dataset, problem_name) -> float:
#     # 文件夹路径
#     folder_path = f'./results/{problem_name}/pops'
#
#     # 用于存储结果的列表
#     res_data = []
#     # 遍历文件夹
#     if os.path.isdir(folder_path):
#         file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
#         sorted_files = sorted(file_list, key=lambda x: int(x.split('=')[-1].split('.')[0]))
#         # for file_name in sorted_files:
#         # print(file_name)
#         file_path = os.path.join(folder_path, sorted_files[-1])
#         # 读取json文件
#         with open(file_path, 'r') as f:
#             json_data = json.load(f)
#             for ind in json_data:
#                 eq_code = ind.get('code')
#                 vaild_nmse, test_nmse = evaluate(dataset["valid_data"], dataset["test_data"], eq_code)
#                 print(vaild_nmse, test_nmse)
#                 test_data = {"valid_data": vaild_nmse, "test_data": test_nmse}

                # json_data.update(test_data)
                # with open(file_path, 'w', encoding='utf-8') as f:
                #     json.dump(json_data, f, indent=5)


def train_evaluate(train_data, param_code):
    try:
        param_sr_module = code_str2code(param_code, "sr2")
        if param_sr_module is None or not hasattr(param_sr_module, "equation"):
            raise ValueError("Generated sr_module is invalid or missing 'equation' method.")
        else:
            y_pred = param_sr_module.equation(*train_data.T[:-1])
            train_nmse = np.mean((train_data[:, -1] - y_pred) ** 2) / np.var(train_data[:, -1])
            return train_nmse
    except Exception as e:
        print("train_evaluate an error occurred:", e)
        return None


def evaluation(file_path, model, dataset, problem_name, save_path) -> float:
    # 文件夹路径
    # folder_path = f'./results/llm_sr/gpt-3.5-turbo/{problem_name}/samples'
    folder_path = file_path
    save_path = save_path
    mk_dir(save_path)

    if os.path.isdir(folder_path):
        file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]
        sorted_files = sorted(file_list, key=lambda x: int(x.split('=')[-1].split('.')[0]))

        for i, file_name in enumerate(sorted_files):
            if i > 3000:
                break
            else:
                if file_name.endswith('.json'):
                    print(file_name)
                    file_path = os.path.join(folder_path, file_name)
                    file_save_path = os.path.join(save_path, file_name)
                    # 读取json文件
                    try:
                        # with open("your_file.json", "r") as f:
                        #     json_data = json.load(f)

                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                            param_code = json_data['param_code']
                            if json_data["objective"] is not None and param_code is not None:

                                train_nmse = train_evaluate(dataset["train_data"], param_code)
                                test_id_nmse, test_ood_nmse = test_evaluate(dataset["valid_data"],
                                                                            dataset["test_data"], param_code)
                                json_data["train_nmse"] = train_nmse
                                json_data["test_id_nmse"] = test_id_nmse
                                json_data["test_ood_nmse"] = test_ood_nmse
                            else:
                                json_data["train_nmse"] = None
                                json_data["test_id_nmse"] = None
                                json_data["test_ood_nmse"] = None
                    except Exception as e:
                        print("train_evaluate an error occurred:", e)
                        json_data = {
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
                            'other_inf': None,
                            'train_nmse': None,
                            'test_id_nmse': None,
                            'test_ood_nmse': None,
                        }

                    with open(file_save_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=4)


if __name__ == '__main__':
    # import pandas as pd

    # problem_names = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    # datas_res = []
    # sample_path = './evosr-llm_results'
    # model = 'gpt-4o-mini'

    # for name in problem_names:
    #     train_data = np.array(pd.read_csv(f"./Problems/llm_sr/{name}" + '/train.csv'))
    #     test_id_data = np.array(pd.read_csv(f"./Problems/llm_sr/{name}" + '/test_id.csv'))
    #     test_ood_data = np.array(pd.read_csv(f"./Problems/llm_sr/{name}" + '/test_ood.csv'))
    #     dataset = {'train_data': train_data, 'valid_data': test_id_data, 'test_data': test_ood_data}

    #     sample_path = f'./evosr-llm_results/{model}/{name}/samples'
    #     save_path = f"./results_full/{model}/{name}/samples"
    #     evaluation(sample_path, model, dataset, name, save_path)

        

    # llm_srbench
    from llm_srbench.llm_srbench_loader import LLMSRBenchLoader
    benchmark_name = "llm_srbench"
    model = "gpt-3.5-turbo"
    # benchmark_name = "llm_sr"
    problem_name = ["bio", "chem", "matsci", "phys"]
    # problem_name = ["matsci", "phys"]
    # problem_name = ["matsci"]
    # problem_name = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    ins_idx_lst = [0, 1]

    for name in problem_name:
        for ins_idx in ins_idx_lst:
            loader = LLMSRBenchLoader(name, ins_idx)
            dataset = loader.get_problem()

            var_name = loader.var_name

            sample_path = f'./evosr-llm_results/{model}/{name}/ins{ins_idx}/samples'
            save_path = f"./results_full/{model}/{name}/ins{ins_idx}/samples"

            evaluation(sample_path, model, dataset, var_name, save_path)
