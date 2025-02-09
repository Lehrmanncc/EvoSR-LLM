import sympy as sp
import time
from sympy import symbols
import ast
import os
import re
import numpy as np
import pandas as pd
from utils.util import code_str2code, get_params_num
from scipy.optimize import minimize


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
        tree = ast.parse(code_str)
        assignments = {}
        return_expr = None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
                value = ast.unparse(node.value) if target else None
                if target not in assignments:
                    assignments[target] = [value]
                else:
                    assignments[target].append(value)
            elif isinstance(node, ast.Return):
                return_expr = ast.unparse(node.value)

        def expand_expr(expr, context):
            changed = True
            max_runtime = 300
            start_time = time.time()

            while changed:
                # 检查是否超时
                if time.time() - start_time > max_runtime:
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
                            expr = expr.replace(var, f"({sub_expr[-1]})")
                            changed = True
                        else:
                            print("error in extracting the core structure of the symbolic expression.")

            return expr

        if return_expr:
            expanded_expr = expand_expr(return_expr, assignments)
            return expanded_expr
        else:
            raise ValueError("The returned expression was not found.")

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


