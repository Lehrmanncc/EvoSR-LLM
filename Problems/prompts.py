import numpy as np
import os
import re


class GetPrompts:
    def __init__(self, problem):
        self.problem = problem
        self.problem_dim = problem.dataset["train_data"][:, :-1].shape[1]
        self.problem_name = problem.problem_name

        (self.prompt_problem, self.prompt_func_input_spec,
         self.prompt_func_output_spec, self.code_example) = self.parse_spec_file()
        # self.prompt_problem, self.prompt_func_input_spec, self.prompt_func_output_spec = self.parse_spec_file()
        # + f"A total of {self.data_num} sample points have been drawn from this dataset. "
        # + f"The range for each independent variable is as follows: {prompt_bound}. ")
        # self.prompt_task = (
        #     "It should be as concise as possible, while fitting the data as accurately as possible."
        #     "Please act as a symbolic regression generator to derive a mathematical expression from the input variables and parameters, and solve the symbolic regression dataset as described above. "
        #     "The goal is to find a symbolic expression that best fits the given data. "
        #     "The symbolic generator should adhere to the following rules and constraints:")
        # self.prompt_problem = self.get_problem_spec()
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

        # self.prompt_func_input_spec = None
        # self.prompt_func_output_spec = None
        self.prompt_func_rule = ["+, -, *, /, ^, sqrt, exp, log, abs", "sin, cos"]

        # if self.prompt_func_output_spec:
        #     self.prompt_inout_inf = (
        #         f"'{self.prompt_func_inputs[:-1]}' are the input variables, {self.prompt_func_input_spec}"
        #         "The list of parameters consists of a series of parameters: 'param[0], param[1], ..., param[m]', where the dimension is defined by you, but you don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. "
        #         f"The output, named '{self.prompt_func_outputs[0]}', {self.prompt_func_output_spec} ")
            # self.prompt_inout_inf = (
            #     f"'{self.prompt_func_inputs[:-1]}' are the input variables, and their dimension depends on the specific dataset."
            #     "The list of parameters consists of a series of parameters: 'param[0], param[1], ..., param[m]', where the dimension is defined by you, but you don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. "
            #     f"The output, named '{self.prompt_func_outputs[0]}', is a mathematical expression involving the input variables and parameters. "
            #     "It should be as concise as possible, while fitting the data as accurately as possible.")

        # self.prompt_inout_inf = (
        #     f"'{self.prompt_func_inputs[:-1]}' are the input variables.  Specifically, {self.prompt_func_input_spec}"
        #     "The list of parameters consists of a series of parameters: 'param[0], param[1], ..., param[m]', where the dimension is defined by you, but you don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. "
        #     f"The output named '{self.prompt_func_outputs[0]}'. Specifically, {self.prompt_func_output_spec}")
        self.prompt_input_inf = (f"Specifically, {self.prompt_func_input_spec} "
                                 f"And '{self.prompt_func_inputs[-1]}' consists of a series of parameters: 'param[0], param[1], ..., param[m]'. You don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. ")

        self.prompt_output_inf = f"Specifically, {self.prompt_func_output_spec} "

        # self.prompt_inout_inf = (
        #     f"Specifically, {self.prompt_func_input_spec}. "
        #     "The list of parameters consists of a series of parameters: 'param[0], param[1], ..., param[m]'. You don't need to determine the exact values. Instead, you should use symbols like 'params[0]' to represent them. "
        #     f"{self.prompt_func_output_spec}")
        # self.prompt_rule_inf = "The symbolic generation rules include basic operators and trigonometric expressions. Note that all operators should be replaced with existing functions from numpy."
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
            with open(
                    os.path.join(path, 'specs', self.problem.benchmark_name,
                                 f'specification_{self.problem_name}.txt'),
                    encoding="utf-8",
            ) as f:
                specification = f.read()

        return specification

    def parse_spec_file(self):
        path = os.path.dirname(os.path.abspath(__file__))
        if self.problem.problem_spec:
            with open(
                    os.path.join(path, 'specs', self.problem.benchmark_name,
                                 f'specification_{self.problem_name}.txt'),
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


if __name__ == '__main__':
    from Problems.problems import ProblemSR
    from utils.util import Paras

    paras = Paras()

    # Set parameters #
    paras.set_paras(operators=['e1', 'e2', 'm1'],
                    problem=['oscillator1'],
                    llm_api_endpoint="gpt.xdmaxwell.top",  # set endpoint
                    llm_api_key=os.environ["XIDIAN_API_KEY"],  # set your key
                    llm_model="gpt-3.5-turbo-1106",  # set llm
                    pop_size=5,
                    n_gen=5,
                    n_process=4,
                    exp_debug_mode=False)

    problem = ProblemSR(paras.problem[0])
    prompt = GetPrompts(problem)
