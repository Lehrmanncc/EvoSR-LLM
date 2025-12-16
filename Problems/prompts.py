import os
import re


class GetPrompts:
    def __init__(self, problem):
        self.problem = problem
        self.problem_dim = problem.dataset["train_data"][:, :-1].shape[1]
        self.problem_name = problem.problem_name

        (self.prompt_problem, self.prompt_func_input_spec,
         self.prompt_func_output_spec, self.code_example) = self.parse_spec_file()

        self.prompt_task = (
            "Your goal is to derive a symbolic expression that best fits the given data. "
            "The expression should be as concise and accurate as possible to prevent overfitting. ")

        self.prompt_func_name = "equation"
        if problem.problem_spec:
            self.prompt_func_inputs = problem.var_name + ['params']
            self.prompt_func_outputs = problem.output_name
        else:
            self.prompt_func_inputs = [f"x{i}" for i in range(self.problem_dim)] + ['params']
            self.prompt_func_outputs = ['y']

        self.prompt_func_rule = ["+, -, *, /, ^, sqrt, exp, log, abs", "sin, cos"]

        self.prompt_input_inf = (f"Specifically, {self.prompt_func_input_spec} "
                                 f"And '{self.prompt_func_inputs[-1]}' consists of a series of parameters: 'param[0], param[1], ..., param[m]'. You don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. ")

        self.prompt_output_inf = f"Specifically, {self.prompt_func_output_spec} "

        input_str = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        self.prompt_other_inf = (f"Note that {input_str}, and '{self.prompt_func_outputs[0]}' are all of type Numpy arrays. "
                                 "The generated expression should be output in standard Numpy notation.")

    def get_problem_spec(self):
        path = os.path.dirname(os.path.abspath(__file__))
        if self.problem.benchmark_name == "Feynman":
            with open(
                    os.path.join(path, 'specs', self.problem.benchmark_name,
                                 f'specification.txt'),
                    encoding="utf-8",
            ) as f:
                specification = f.read()
        else:
            # 对于llm_srbench，根据ins_idx读取不同的specification文件
            if self.problem.benchmark_name == "llm_srbench" and self.problem.ins_idx is not None:
                spec_filename = f'specification_{self.problem_name}{self.problem.ins_idx}.txt'
            else:
                spec_filename = f'specification_{self.problem_name}.txt'
            
            with open(
                    os.path.join(path, 'specs', self.problem.benchmark_name, spec_filename),
                    encoding="utf-8",
            ) as f:
                specification = f.read()

        return specification

    def parse_spec_file(self):
        path = os.path.dirname(os.path.abspath(__file__))
        if self.problem.problem_spec:
            # 对于llm_srbench，根据ins_idx读取不同的specification文件
            if self.problem.benchmark_name == "llm_srbench" and self.problem.ins_idx is not None:
                spec_filename = f'specification_{self.problem_name}{self.problem.ins_idx}.txt'
            else:
                spec_filename = f'specification_{self.problem_name}.txt'
            
            with open(
                    os.path.join(path, 'specs', self.problem.benchmark_name, spec_filename),
                    encoding="utf-8") as file:
                content = file.read()
        # Split the content by section headers
        sections = {
            "Problem Task": None,
            "Equation Input Specification": None,
            "Equation Output Specification": None,
            "Equation Code Example": None,
        }

        # Regex to match sections
        for section in sections.keys():
            match = re.search(rf"### {section}\n(.*?)(?=\n###|$)", content, re.DOTALL)
            if match:
                sections[section] = match.group(1).strip() or None

        # Map the sections to the corresponding variables
        prompt_problem = sections["Problem Task"]
        prompt_func_input_spec = sections["Equation Input Specification"]
        prompt_func_output_spec = sections["Equation Output Specification"]
        prompt_code_example = sections["Equation Code Example"]

        return prompt_problem, prompt_func_input_spec, prompt_func_output_spec, prompt_code_example

