# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
import sympy as sp
import time
from sympy import symbols
import ast
import os
import re
import numpy as np
import pandas as pd
from utils.util import code_str2code, get_var_num, get_params_num
from scipy.optimize import minimize


# class Probs():
#     def __init__(self, paras):
#
#         if not isinstance(paras.dataset, str):
#             self.prob = paras.dataset
#             print("- Prob local loaded ")
#         elif paras.dataset == "tsp_construct":
#             from .optimization.tsp_greedy import run
#             self.prob = run.TSPCONST()
#             print("- Prob " + paras.dataset + " loaded ")
#         elif paras.dataset == "bp_online":
#             from .optimization.bp_online import run
#             self.prob = run.BPONLINE()
#             print("- Prob " + paras.dataset + " loaded ")
#         else:
#             print("dataset " + paras.dataset + " not found!")
#
#     def get_problem(self):
#
#         return self.prob


class ProblemSR:
    def __init__(self, benchmark_name, problem_name, problem_spec=True):
        self.benchmark_name = benchmark_name
        self.problem_name = problem_name
        self.var_name = None
        self.output_name = None
        self.problem_spec = problem_spec
        if not isinstance(problem_name, str):
            self.dataset = problem_name
            print("- Problem local loaded -")
        elif isinstance(problem_name, str):
            self.dataset = self.get_problem()
            print("- Problem " + problem_name + " loaded ")
        else:
            print("dataset " + problem_name + " not found!")

        self.fe = 0

    def get_problem(self):
        file_path = os.path.abspath(__file__)
        problem_path = os.path.join(os.path.dirname(file_path), self.benchmark_name, self.problem_name)
        data = pd.read_csv(problem_path + '/train.csv')
        self.var_name, self.output_name = data.columns.tolist()[:-1], data.columns.tolist()[-1]
        train_data = np.array(data)

        if self.benchmark_name == "Feynman":
            test_data = np.array(pd.read_csv(problem_path + '/test.csv'))
            dataset = {'train_data': train_data, 'test_data': test_data}
        else:
            valid_data = np.array(pd.read_csv(problem_path + '/test_id.csv'))
            test_data = np.array(pd.read_csv(problem_path + '/test_ood.csv'))
            dataset = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data}
        return dataset

    def extract_expression(self, code_str):
        """
        从代码字符串中提取核心表达式。
        """
        tree = ast.parse(code_str)
        assignments = {}
        return_expr = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):  # 查找赋值语句
                target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                value = ast.unparse(node.value) if target else None
                if target not in assignments:
                    assignments[target] = [value]
                else:
                    assignments[target].append(value)
            elif isinstance(node, ast.Return):  # 查找 return 语句
                return_expr = ast.unparse(node.value)

        # 展开表达式，确保所有变量都被替换成最终的表达式
        def expand_expr(expr, context):
            changed = True
            max_runtime = 300  # 最大运行时间，单位为秒
            start_time = time.time()  # 记录开始时间

            while changed:
                # 检查是否超时
                if time.time() - start_time > max_runtime:
                    print("超过最大运行时间，停止替换以避免死循环。")
                    expr = None
                    break
                changed = False

                for var, sub_expr in context.items():
                    if re.search(r'\b' + re.escape(var) + r'\b', expr):
                        if len(sub_expr) > 1:
                            # 替换为最后一个赋值表达式
                            expr = expr.replace(var, f"({sub_expr[-1]})")
                            sub_expr.pop(-1)
                            changed = True
                        elif len(sub_expr) == 1:
                            # 替换为最后一个赋值表达式
                            expr = expr.replace(var, f"({sub_expr[-1]})")
                            changed = True
                        else:
                            print("计算表达式复杂度中，提取符号表达式主干错误")

            return expr

        if return_expr:
            expanded_expr = expand_expr(return_expr, assignments)
            return expanded_expr
        else:
            raise ValueError("未找到返回的表达式")

    def parse_expression(self, expr_str):
        replacements = {
            "np.log": "log",
            "np.sin": "sin",
            "np.cos": "cos",
            "np.tan": "tan",
            "np.exp": "exp",
            "np.sqrt": "sqrt",
            "np.abs": "Abs",
            "np.tanh": "tanh",
            "np.cosh": "cosh",
            "np.sinh": "sinh",
            "np.arcsin": "asin",
            "np.arccos": "acos",
            "np.gradient": "diff",
            "np.pi": "pi"
        }

        sym_dict = {var: symbols(var) for var in self.var_name}
        for np_func, sp_func in replacements.items():
            expr_str = expr_str.replace(np_func, sp_func)
        return sp.sympify(expr_str, sym_dict, evaluate=False)

    def cal_equ_complexity(self, code_str):
        try:
            expr_str = self.extract_expression(code_str)
            if expr_str is not None:
                parse_expr = self.parse_expression(expr_str)
                c = sum(1 for _ in sp.preorder_traversal(parse_expr))
                return c
            else:
                return None
        except Exception as e:
            print(e)
            return None

    def train_evaluate(self, sr_code_str):
        self.fe += 1
        train_data = self.dataset["train_data"]
        try:
            sr_module = code_str2code(sr_code_str)
            loss_func = lambda params: self.train_eval(train_data, sr_module, params)

            params_init = [1.0] * get_params_num(sr_code_str)
            res = minimize(loss_func, params_init, method="BFGS")
            opt_params = res.x

            score = self.train_eval(train_data, sr_module, opt_params)
            if np.isnan(score):
                return None, None
            else:
                return score, opt_params
        except Exception as e:
            print("train_evaluate an error occurred:", e)

            return None, None

    def valid_evaluate(self, sr_code_str):
        valid_data = self.dataset["valid_data"]
        try:
            sr_module = code_str2code(sr_code_str)
            nmse = self.test_eval(valid_data, sr_module)
            return nmse

        except Exception as e:
            print("An error occurred:", e)
            return None

    def test_evaluate(self, sr_code_str):
        test_data = self.dataset["test_data"]
        try:
            sr_module = code_str2code(sr_code_str)
            nmse = self.test_eval(test_data, sr_module)
            return nmse

        except Exception as e:
            print("An error occurred:", e)
            return None

    def train_eval(self, train_data, sr_code, params):
        y_pred = sr_code.equation(*train_data.T[:-1], params)
        return np.mean((y_pred - train_data[:, -1]) ** 2)

    def test_eval(self, test_data, sr_code):
        y_pred = sr_code.equation(*test_data.T[:-1])
        nmse = np.mean((test_data[:, -1] - y_pred) ** 2) / np.var(test_data[:, -1])
        return nmse
        # return np.mean(y_pred - test_data[:, -1]) ** 2


if __name__ == "__main__":
    from utils.util import Paras


    paras = Paras()

    # Set parameters #
    paras.set_paras(method="eoh",
                    ec_operators=['e1', 'e2', 'm1', 'm2', 'm3'],  # operators in EoH
                    problem="nguyen",  # ['tsp_construct','bp_online','tsp_gls','fssp_gls']
                    llm_api_endpoint="gpt.xdmaxwell.top",  # set endpoint
                    llm_api_key=os.environ["XIDIAN_API_KEY"],  # set your key
                    llm_model="gpt-3.5-turbo-1106",  # set llm
                    ec_pop_size=4,
                    ec_n_pop=2,
                    exp_n_proc=4,
                    exp_debug_mode=False)

    def extract_expression(code_str):
        """
        从代码字符串中提取核心表达式。
        """
        tree = ast.parse(code_str)
        assignments = {}
        return_expr = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):  # 查找赋值语句
                target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                value = ast.unparse(node.value) if target else None
                if target and value:
                    assignments[target] = value
            elif isinstance(node, ast.Return):  # 查找 return 语句
                return_expr = ast.unparse(node.value)

        # 展开表达式，确保所有变量都被替换成最终的表达式
        def expand_expr(expr, context):
            for var, sub_expr in reversed(context.items()):
                expr = expr.replace(var, f"({sub_expr})")
            return expr

        if return_expr:
            expanded_expr = expand_expr(return_expr, assignments)
            return expanded_expr
        else:
            raise ValueError("未找到返回的表达式")



    problem = ProblemSR("llm_sr", 'oscillator1')
    # code_string = """import numpy as np\ndef equation(x1, x2, params):\n    return params[0] * x1**3 + params[1] * x2**2 + params[2] * x1"""
    # code_string = "import numpy as np\n\ndef equation(x, v):\n    damping_coeff = 0.00015\n    custom_param = -1e-05\n    new_param = -1e-05\n    \n    a = 0.00044 * np.exp(0.70885 * np.abs(v)) + -0.00011 * np.cosh(0.074 * np.log(np.abs(x) * np.abs(v))) + -0.0 * np.sqrt(np.abs(np.sin(x * v))) + -0.00078 * v**2 * np.tanh(np.arcsin(x + v) * np.arccos(x + v)) + 2e-05 * (np.sin(x) + np.cos(v)) + -0.00029 * np.exp(-3.04366 * np.abs(x)) + -0.17605 * x + -0.00137 * v + -0.03716 * np.sin(x) + -0.50008 * x * v + 0.87466 * v**5 + -0.54868 * v**3 - -1e-05 * np.log(np.abs(v)**3) + 0.13848 * x**3 + 5e-05 * np.exp(3.7218 * np.sin(x) * np.cos(v)) + 0.00027 * np.sin(v)**2 + 2e-05 * (np.sin(x)**2 * np.cos(v)**2) + 2e-05 * np.arccos(x)**2 * np.tan(v)**2 - 0.00075 * np.log(np.abs(x)**3) * np.exp(np.abs(v)) * np.sin(x) * np.cos(v) + 0.00305 * np.cos(x) * np.sin(v) * np.sqrt(np.abs(x)) + 0.02144 * np.sqrt(np.abs(x)) * np.exp(0.08508 * np.abs(x)) * np.tanh(x) + -0.00054 * np.exp(-0.21672 * np.abs(x)) - -0.00048 * np.exp(0.31767 * np.abs(v)) + 2e-05 * np.tanh(x * v) + 1e-05 * np.sin(x + v) * np.cos(x + v) + 0.0 * np.tanh(x + v) + 0.0 * np.tanh(x + v) + -1e-05 * v + 0.0 * np.tanh(x + v) + -3e-05 * v + -1e-05 * np.cos(x)**2 + 0.00027 * np.sin(v)**2 + 0.00013 * np.log(np.abs(v)) * np.sqrt(np.abs(v)) + 1e-05 * np.sin(np.log(np.abs(x))) + 1e-05 * np.sin(1.0 + 1.0 * np.log(np.abs(x))) + np.cos(0.78539 + 0.78539) * np.sqrt(np.abs(v)) + 7e-05 * v**2 * np.log(np.abs(x)) + 1e-05 * v**2 * np.log(np.abs(x)) * np.log(np.abs(x)) + -2e-05 * np.cos(np.pi * x) * np.sin(np.pi * v) + -0.0003 + -5e-05 * np.log(np.abs(v)) * np.abs(x)**2 + 0.00014 * np.sqrt(np.abs(v)) * x**3 + 0.00071 * np.log(np.abs(v)) * v**3 + 1e-05 * v**2 * np.log(np.abs(x)) * np.log(np.abs(x)) + -1e-05 * np.log(np.abs(v)) * x**4 + damping_coeff * x * v + custom_param * np.log(np.abs(x + v)**2) + new_param * np.log(np.abs(x)) * np.log(np.abs(v)) + new_param * np.cos(x * v * 1.0049) + new_param * np.sin(x * v * 1.013) + new_param * np.tanh(x + v) + custom_param * np.cos(x) * np.sinh(v) + new_param * np.log(np.abs(x)) * np.sqrt(np.abs(v)) + new_param * np.log(np.abs(v)) * np.abs(x)**2\n    \n    return a"
    # code_string = "import numpy as np\n\ndef equation(x, v):\n    a = 0.7065 * x + 0.00288 * v + -0.9045 * np.sin(x) + -0.49901 * x * v + -0.56958 * v**3\n    return a"
    # code_string = "import numpy as np\n\ndef equation(x, v):\n    a = 0.00033 * np.exp(0.9138 * np.abs(x)) + -0.1681 * x + 0.001 * v + -0.02799 * np.sin(x) + -0.49964 * x * v + 0.87037 * v**5 + -0.573 * v**3 - -0.0 * np.log(np.abs(v)**3) + 0.14298 * x**3 + -0.00041 * np.exp(1.01747 * np.sin(x) * np.cos(v)) + 0.00045 * np.log(np.abs(x)**3) * np.exp(np.abs(v)) * np.sin(x) * np.cos(v)\n    return a"
    # code_string = "import numpy as np\n\ndef equation(x, v):\n    a = params[0] * np.exp(params[1] * np.abs(x)) + params[2] * x + params[3] * v + params[4] * np.sin(x) + params[5] * x * v + params[6] * v**5 + params[7] * v**3 - params[8] * np.log(np.abs(v)**3) + params[9] * x**3 + params[10] * np.exp(params[11] * np.sin(x) * np.cos(v)) + params[12] * np.log(np.abs(x)**3) * np.exp(np.abs(v)) * np.sin(x) * np.cos(v)\n    return a"
    # code_string = "import numpy as np\n\ndef equation(x, v):\n    \"\"\"\n    Calculate the acceleration of a damped nonlinear oscillator system \n    with a novel combined term considering the interplay of position, velocity,\n    and system nonlinearity.\n\n    Parameters:\n    x (numpy array): Current position observations.\n    v (numpy array): Current velocity observations.\n    params (numpy array): Array containing parameters \n                          [param[0], param[1], ..., param[m]].\n\n    Returns:\n    numpy array: Calculated acceleration.\n    \"\"\"\n    # Unpack parameters for clarity\n    param_0 = -0.45644\n    param_1 = -0.11926\n    param_2 = 1.0402\n    param_3 = 0.33502\n    param_4 = 0.7417\n    param_5 = -0.07\n    param_6 = -0.19534\n    param_7 = 0.84182\n    param_8 = 0.81051\n    param_9 = -0.12979\n    param_10 = 1.49613\n    param_11 = -0.5654\n    param_12 = 0.04699\n\n    # Novel combined term considering interplay of position, velocity, and nonlinearity\n    combined_term = (param_0 * x * v + \n                     param_1 * np.abs(v) * np.exp(param_2 * x) +\n                     param_3 * (x ** 2) * np.abs(v) * np.exp(param_4 * (x + v)) +\n                     param_5 * np.cos(param_6 * x + param_7) * np.exp(param_8 * (x * v)) +\n                     param_9 * np.sin(param_10 * x + param_11) * np.exp(param_12 * (x - v)))\n\n    # Calculate acceleration incorporating the novel term\n    a = combined_term\n    \n    return a"
    # code_string = "import numpy as np\n\ndef equation(t, x, v):\n    param0 = -6e-05  # Parameter for scaling the restoring force\n    param1 = 0.0  # Parameter for scaling the damping\n    param2 = 0.48643  # Parameter for additional nonlinearity\n    \n    damping_coefficient = -param1 * np.cos(param2 * t)  # Time-dependent damping coefficient\n    restoring_coefficient = param0 * x**3 + param1 * x**2  # Nonlinear restoring force\n    adaptation_factor = -1e-05 * x * v  # Adaptation factor based on position and velocity\n    nonlinear_damping_coefficient = -1e-05 * v * x  # Nonlinear damping coefficient\n    frequency_dependent_damping = -499.99202  # Frequency-dependent damping coefficient\n    frequency_feedback = -0.0 * np.sin(1.90435 * t)  # Frequency-dependent feedback term\n    \n    adaptive_damping = (\n        damping_coefficient * np.abs(v) +\n        adaptation_factor +\n        nonlinear_damping_coefficient +\n        frequency_dependent_damping * np.gradient(v, edge_order=2) +\n        frequency_feedback\n    )\n\n    a = -adaptive_damping + restoring_coefficient\n\n    return a"
    # code_string = "import numpy as np\n\n# {Bacterial growth in E. coli is often modeled using the Monod equation, which relates the growth rate to substrate concentration. This equation shows how nutrient availability can limit growth and is influenced by environmental factors such as temperature and pH.}\n# {To improve the model, incorporate an interaction term between substrate concentration and pH, as variations in pH could impact the bioavailability of the substrate, thus altering the growth rate more significantly than when these factors are considered independently.}\n\ndef equation(b, s, temp, pH):\n    r = (0.0 * b * s / (0.62365 + s)) * np.exp(0.11771 * (temp - -129.56826)) * (1 / (1 + (pH - -153.97336) ** 2))\n    return r"
    code_string = "import numpy as np\n\ndef equation(b, s, temp, pH):\n    log_b = np.log1p(b)  # Logarithm of population density\n    sqrt_s = np.sqrt(s)  # Square root of substrate concentration\n\n    # Enhanced pH effect: linear and exponential components based on pH\n    pH_effect = -0.00197 * (pH - 7) + np.exp(-1.02107 * (pH - 7) ** 2)\n\n    # Enhanced temperature effect: linear and exponential components \n    temp_effect = 1 + 0.85896 * (temp - 0.57688) + np.exp(-0.91457 * (temp - 1.17736))\n\n    # Calculate growth rate, incorporating enhanced pH and temperature effects\n    r = 0.41288 * (log_b ** 0.87652) * (sqrt_s ** 2) / (1 + 1.23801 * s) * pH_effect / temp_effect\n\n    # Implement a new mathematical function on the output for increased complexity\n    new_function = 0.79864 * np.sin(0.84168 * r)\n    \n    # Introducing an exponential function on the output growth rate\n    exponential_function = np.exp(-0.99002 * r)\n    \n    # Combine the new function and exponential function with the growth rate\n    r = r + new_function\n    r = r * exponential_function\n\n    return r"
    print(code_string)


    expr_str = problem.extract_expression(code_string)
    # expr_str = "import numpy as np\n\ndef equation(strain: np.ndarray, temp: np.ndarray) -> np.ndarray:\n    \"\"\" Mathematical function for stress in Aluminium rod\n\n    Args:\n        strain: A numpy array representing observations of strain.\n        temp: A numpy array representing observations of temperature.\n        params: Array of numeric constants or parameters to be optimized\n\n    Return:\n        A numpy array representing stress as the result of applying the mathematical function to the inputs.\n    \"\"\"\n    \"\"\"Enhances the model by introducing additional parameters and expressions for stress calculation.\"\"\"\n    \n    E0 = 36.697735439939045  # Initial Young's modulus at reference temperature (E_0)\n    sigma_y0 = 0.8516295001204223  # Initial yield strength at reference temperature (sigma_y0)\n    n = 2.6971478465712697  # Work hardening exponent (n)\n    alpha = 0.1058760199085317  # Linear coefficient of thermal expansion (alpha)\n    beta = 1.1783312258568164  # Nonlinear coefficient affecting yield strength (beta for temperature effect)\n    T_ref = 0.3849165912445739  # Reference temperature (T_ref)\n    T_max = 1.0  # Maximum temperature to avoid excessive yield strength reduction (T_max)\n    k = -0.2810064757435788  # Additional coefficient for stress relaxation with temperature\n    c = 1.0  # Coefficient for strain rate dependence\n    A = 1.000673438407918  # Activation energy for temperature effects on yield strength\n\n    E = E0 * (1 + alpha * (temp - T_ref))  # Temperature-adjusted Young's modulus\n    \n    # Nonlinear adjustment of yield strength based on temperature and strain rate\n    yield_strength_temp_strainrate = sigma_y0 * (1 - beta * np.clip(temp - T_ref, 0, T_max)) * np.exp(A / (temp + 273.15))\n\n    elastic_limit = yield_strength_temp_strainrate / E  # Threshold for plastic deformation\n    \n    stress = np.zeros_like(strain)  # Initialize stress array\n\n    elastic_condition = strain < elastic_limit  # Condition for elastic and plastic regions\n    \n    # Calculate stress based on material behavior in elastic and plastic regions\n    stress = np.where(\n        elastic_condition,\n        E * strain,  # Elastic region: Hooke's Law\n        yield_strength_temp_strainrate + ((strain - elastic_limit) ** n * k)  # Plastic region: Work hardening model adjusted by relaxation factor\n    )\n\n    stress = np.maximum(stress, 0)  # Ensure stress is non-negative\n\n    return stress  # Return computed stress values based on the model improvements.\n\n"
    print(expr_str)
    # parse_expr = problem.parse_expression(expr_str)
    # print(parse_expr)
    # c = sum(1 for _ in sp.preorder_traversal(parse_expr))
    # print(c)
    # parse_expr = parse_expression(expr_str, ["x", "v"])

    # c1 = problem.cal_equ_complexity(code_string)
    # print(c1)
    # score, opt_params = problem.train_evaluate(code_string)
    # print(opt_params)
    # r = problem.test_evaluate(problem.dataset[0], code_string, opt_params)
    # print(r)
