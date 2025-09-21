import ast
import json
import glob
from typing import List, Optional, Tuple

from Problems.problems import ProblemSR
import warnings
from scipy.optimize import minimize
import re
import time
import sympy as sp
from sympy import symbols
import ast
import types
import sys

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
            if target not in assignments:
                assignments[target] = [value]
            else:
                assignments[target].append(value)
        elif isinstance(node, ast.Return):  # 查找 return 语句
            return_expr = ast.unparse(node.value)

    def expand_expr(expr, context):
        changed = True
        max_runtime = 15  # 最大运行时间，单位为秒
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


def parse_expression(expr_str, var_name):
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
        "np.pi": "pi",
        "np.arctan": "atan",
    }

    sym_dict = {var: symbols(var) for var in var_name}
    for np_func, sp_func in replacements.items():
        expr_str = expr_str.replace(np_func, sp_func)
    return sp.sympify(expr_str, sym_dict, evaluate=False)


def cal_equ_complexity(code_str, var_name):
    try:
        expr_str = extract_expression(code_str)
        if expr_str is not None:
            parse_expr = parse_expression(expr_str, var_name)
            c = sum(1 for _ in sp.preorder_traversal(parse_expr))
            return c
        else:
            return None
    except Exception as e:
        print(e)
        return None

def _extract_var_names_from_func(code: str) -> List[str]:
    """Parse function args from a def equation signature (exclude 'params')."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args if a.arg != "params"]
            return names
    return []


def main():
    datasets = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
    results: List[Tuple[str, str, Optional[int]]] = []

    for ds in datasets:
        pattern = f"baselines_results/drsr/gpt-3.5-turbo/{ds}/**/best.json"
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(path, "r") as f:
                    j = json.load(f)
                code = j.get("optimized_equation") or j.get("original_equation")
                if not code:
                    results.append((ds, path, None))
                    continue
                var_names = _extract_var_names_from_func(code)

                c = cal_equ_complexity(code, var_names)
                results.append((ds, path, c))
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append((ds, path, None))

    # Pretty print
    for ds, path, c in results:
        print(f"{ds}\t{path}\tcomplexity={c}")


if __name__ == "__main__":
    main()
