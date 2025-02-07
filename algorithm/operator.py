import copy
import re
import numpy as np
import time
from llm.interface_LLM import InterfaceLLM
from Problems.prompts import GetPrompts


class OperatorPrompt(GetPrompts):
    def __init__(self, problem):
        super().__init__(problem)
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ",".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        if len(self.prompt_func_rule) > 1:
            self.joined_rule = "and ".join("'" + s + "'" for s in self.prompt_func_rule)
        else:
            self.joined_rule = "'" + self.prompt_func_rule[0] + "'"

        # self.inint_operator1_prompt = "Regarding the parent equation at the code level, please help me create a new equation that preserves the core structure of the parent equations' code while incorporating localized adjustments."
        # self.inint_operator2_prompt = "Regarding the parent equation at the code level, please help me create a new equation that totally different form from the parent equations' code."
        # self.inint_operator3_prompt = "Regarding the parent equation at the code level, please help me create a new equation that extracts the core components of the parent equations' code and combines them to form new ones."
        self.inint_operator1_prompt = "Regarding the parent equation at the knowledge level, please help me create a new equation by integrating the scientific knowledge contained in the parent equation."
        self.inint_operator2_prompt = "Regarding the parent equation at the insight level, please help me create a new equation that fundamentally differs in form from the insight of the parent equation."
        self.inint_operator3_prompt = "Regarding the parent equation at the code level, please help me create a new equation by extracting the core components from the parent equation’s code and combining them to form a novel expression."

        self.operator_prompt_pop = [self.inint_operator1_prompt, self.inint_operator2_prompt,
                                    self.inint_operator3_prompt]
        self.example = ("The following example equation code is provided for formatting reference:\n"
                        + self.code_example)
        # self.example = ("The following example equation code is provided for formatting reference:\n"
        #                   + '```python\nimport numpy as np\n\ndef equation(x1, x2, params):\n    y = params[0] * x1 + params[1] * x2\n    return y\n ```\n'
        #                   + "Please generate a new equation using the same formatting, but do not directly replicate the example.")

    def get_problem_task(self):
        prompt_content = (self.prompt_problem + "\n"
                          + self.prompt_task + "\n"
                          + "The basic operations and trigonometric expressions used in the equation are"
                          + self.joined_rule + ". " + "\n"
                          + "You should adhere to the following rules and constraints:")
        return prompt_content

    def construct_instruction(self):
        knowledge_prompt = "1. Summarize one piece of prior knowledge in the current field, ensuring the description is concise and clear, and must be enclosed within a single pair of curly braces."
        insight_prompt = "2. Based on the prior knowledge, propose an innovative insight to improve the equation. Ensure the description is concise and clear, and must be enclosed within a single pair of curly braces."
        # knowledge_prompt = (
        #     "Summarize 1-2 key theoretical concepts that the equation might involve in a concise and accurate manner. "
        #     "Ensure the description of scientific knowledge is enclosed in a single pair of curly braces.")
        # insight_prompt = (
        #     "Then, analyze the fundamental principles of these theories and think beyond the existing framework. "
        #     "Based on logical reasoning, propose a bold and innovative scientific insight regarding the equation. "
        #     "Ensure the description of the scientific insight is enclosed in a single pair of curly braces.")
        # code_prompt = (
        #             "Finally, based on the proposed scientific insight, implement the equation in Python as a function named"
        #             + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
        #             + " input(s): " + self.joined_inputs + ". The function should return "
        #             + str(len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " + "\n"
        #             + self.prompt_inout_inf + " " + self.prompt_other_inf)
        code_prompt = (
                "3. Finally, based on the proposed scientific insight, implement the equation in Python as a function named"
                + self.prompt_func_name + ". This function should accept " + str(len(self.prompt_func_inputs))
                + " input(s): " + self.joined_inputs + ". " + self.prompt_input_inf + "The function should return "
                + str(len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". "
                + self.prompt_output_inf + "\n" + self.prompt_other_inf)

        instruction_prompt = knowledge_prompt + "\n" + insight_prompt + "\n" + code_prompt
        return instruction_prompt

    def get_init_prompt(self):
        task_prompt = self.get_problem_task()
        instruction_prompt = self.construct_instruction()
        return task_prompt + "\n" + instruction_prompt + "\n" + self.example

    def get_parent_prompt(self, parents):
        prompt_parents = ""
        for i in range(len(parents)):
            prompt_parents = prompt_parents + "No." + str(
                i + 1) + "  equation knowledge, equation insight and equation code are: \n" + \
                             parents[i]['knowledge'] + "\n" + parents[i]['insight'] + "\n" + parents[i]['code'] + "\n"
        return prompt_parents

    def get_prompt_operator(self, parents, operator):
        prompt_parents = self.get_parent_prompt(parents)

        task_prompt = self.get_problem_task()
        instruction_prompt = self.construct_instruction()
        prompt_content = (task_prompt + "\n"
                          + "I have " + str(len(parents)) + " existing equations with their knowledge, insight and codes as follows: \n"
                          + prompt_parents + "\n"
                          + operator + "\n"
                          + instruction_prompt)

        return prompt_content


class MetaPrompt:
    def __init__(self):

        self.MKP = ("Generate a new prompt about performing an operation on the scientific knowledge from  parent equation, such as reorganizing, expanding, reinterpreting, or any other novel operation. "
                    "Make the prompt as diverse as possible, ensuring it is a declarative statement. "
                    "Finally, ensure that the generated prompt entry does not exceed one.")
        # and must begin with 'Please help me create a new equation that...'
        self.MIP = ("Generate a new prompt about performing an operation on the scientific insight from parent equation, such as reorganizing, expanding, reinterpreting, or any other novel operation. "
                    "Make the prompt as diverse as possible, ensuring it is a declarative statement. "
                    "Finally, ensure that the generated prompt entry does not exceed one.")
        self.MCP = ("Generate a new prompt about performing an operation on the code from parent equation, such as modifying, simplifying, restructuring, or introducing a novel operation. "
                    "Make the prompt as diverse as possible, ensuring it is a declarative statement. "
                    "Finally, ensure that the generated prompt entry does not exceed one.")
        self.task = ("You are a helpful prompt generator. "
                     "Your goal is to produce effective prompts that guide the evolution of equations, where the equations are selected parent equations from symbolic regression tasks. "
                     "Each equation consists of three parts: the involved scientific knowledge, the proposed scientific insight, and the corresponding implementation code."
                     "You must follow the rules and constraints below:")

        self.example = ("The following example prompt is provided for formatting reference:\n"
                        + "'please help me create a new equation that extracts the core components of the parent equations' code and combines them to form new ones.'")
        #                   + '```python\nimport numpy as np\n\ndef equation(x1, x2, params):\n    y = params[0] * x1 + params[1] * x2\n    return y\n ```\n'
        #                   + "Please generate a new equation using the same formatting, but do not directly replicate the example.")

    def construct_mkp(self):
        meta_prompt = self.task + "\n" + self.MKP
        return meta_prompt

    def construct_mip(self):
        meta_prompt = self.task + "\n" + self.MIP
        return meta_prompt

    def construct_mcp(self):
        meta_prompt = self.task + "\n" + self.MCP
        return meta_prompt

    def check_prompt(self, op_label):
        if op_label == "MKP":
            return self.construct_mkp()
        elif op_label == "MIP":
            return self.construct_mip()
        elif op_label == "MCP":
            return self.construct_mcp()
        else:
            print("OP LABEL ERROR!")


class Operator(OperatorPrompt):
    def __init__(self, api_endpoint, api_key, model_LLM,
                 llm_use_local, llm_local_url, problem):
        # set LLMs
        super().__init__(problem)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.meta_prompt = MetaPrompt()

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM,
                                          llm_use_local, llm_local_url)

    def llm_get_equ(self, prompt_content):
        print("进入llm_get_equ: ")
        response = self.interface_llm.get_response(prompt_content)
        # print(response)

        # 匹配大括号 {} 中的所有内容(包括换行符)，得到llm输出当前算法的解释
        equ_desc = re.findall(r"\{(.*?)\}", response, re.DOTALL)
        # print(equ_desc)
        if len(equ_desc) == 0:
            if 'python' in response:
                # 匹配字符串开头到python之前的内容
                equ_desc = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                equ_desc = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                equ_desc = re.findall(r'^.*?(?=def)', response, re.DOTALL)
        # 匹配字符串从 "import" 开始，到 "return" 结束
        equ_code = re.findall(r"import.*return", response, re.DOTALL)
        if len(equ_code) == 0:
            equ_code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        # 如果llm返回的算法介绍与code都是空，则再重复尝试3次调用llm输出
        while (len(equ_desc) == 0 or len(equ_code) == 0):
            print("Error: equ_desc or equ_code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            equ_desc = re.findall(r"\{(.*?)\}", response, re.DOTALL)
            if len(equ_desc) == 0:
                if 'python' in response:
                    equ_desc = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    equ_desc = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    equ_desc = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            equ_code = re.findall(r"import.*return", response, re.DOTALL)
            if len(equ_code) == 0:
                equ_code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        equ_knowledge = equ_desc[0]
        equ_insight = equ_desc[1]
        equ_code = equ_code[0]

        # 添加return 后的输出
        equ_code = equ_code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [equ_knowledge, equ_insight, equ_code]

    def llm_get_op_prompt(self, op_label):
        metaprompt_content = self.meta_prompt.check_prompt(op_label)
        response = self.interface_llm.get_response(metaprompt_content)
        n_retry = 1
        # 如果llm返回的response都是空，则再重复尝试3次调用llm输出
        while len(response) == 0:
            print("Error: equ_desc or equ_code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(metaprompt_content)

            if n_retry > 3:
                break
            n_retry += 1

        prompt_op = copy.deepcopy(response)

        return prompt_op

    def init_equ(self):
        prompt_content = self.get_init_prompt()
        [knowledge, insight, code] = self.llm_get_equ(prompt_content)

        return [knowledge, insight, code]

    def op_offspring(self, parent, operator_prompt):
        print("获得输入prompt")
        prompt_content = self.get_prompt_operator(parent, operator_prompt)
        print("得到输入prompt，输入llm得到子代")
        [knowledge, insight, code] = self.llm_get_equ(prompt_content)
        return [knowledge, insight, code]

    def add_new_op(self, op_label):
        prompt_op = self.llm_get_op_prompt(op_label)
        if op_label == "MKP":
            prompt_op = "Regarding the parent equation at the knowledge level. " + prompt_op
        elif op_label == "MIP":
            prompt_op = "Regarding the parent equation at the insight level. " + prompt_op
        elif op_label == "MCP":
            prompt_op = "Regarding the parent equation at the code level. " + prompt_op

        return prompt_op


if __name__ == "__main__":
    import os
    from Problems.prompts import GetPrompts
    from Problems.problems import ProblemSR
    from utils.util import Paras

    paras = Paras()

    # Set parameters #
    paras.set_paras(operators=['e1', 'e2', 'm1'],
                    benchmark="llm_sr",
                    problem=["stressstrain"],
                    # llm_api_endpoint="gpt.xdmaxwell.top",  # set endpoint
                    llm_api_endpoint="api.vveai.com",
                    # llm_api_key=os.environ["XIDIAN_API_KEY"],  # set your key
                    llm_api_key="sk-dDK1z0C2qtgaysRB2192C116AaAd406f96532f7e695a567c",
                    llm_model="gpt-3.5-turbo",  # set llm
                    pop_size=5,
                    offspring_size=2,
                    max_fe=200,
                    n_process=4,
                    exp_debug_mode=False)

    problem = ProblemSR(paras.benchmark, paras.problem[0])
    operator_prompt = OperatorPrompt(problem)
    prompt = operator_prompt.get_init_prompt()
    print(prompt)
    # content = operator_prompt.get_prompt_i1()
    # operator = Operator(paras.llm_api_endpoint, paras.llm_api_key, paras.llm_model, None, None, problem)
    # prompt, label = operator.llm_get_op_prompt()
    # print(prompt, label)
