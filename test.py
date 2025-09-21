import json

import pandas as pd
from scipy.stats import rankdata
import numpy as np
import os
import matplotlib.pyplot as plt

# print(0.1 * np.exp(-151 / 40))
# print(0.1 * np.exp(-40 / 40))

import numpy as np

import numpy as np

def GA(params1, params2, params3):
    ...
    return

def equation(b, s, temp, pH, params):
    # Substrate effect with saturation
    substrate_effect = s / (s + params[0])
    # Polynomial interaction for temperature and pH
    temp_pH_interaction = (temp ** params[1]) * (pH ** params[2])
    # Combined dynamic coefficient considering interactions between substrate,
    # population density, and temperature-pH
    dynamic_coefficient = params[3] * substrate_effect * b * temp_pH_interaction
    # Interaction damping based on deviations from optimal temperature and pH
    interaction_damping = ((params[4] * dynamic_coefficient)
                           / (1 + np.abs(temp - params[5]) + np.abs(pH - params[6])) ** params[7])
    # Combined effect considering previous interaction damping and additional parameters
    combined_effect = params[8] * interaction_damping
    # Exponential decay based on distance from optimal temperature and pH
    decay_effect = (np.exp(-np.abs(temp - 37) / params[9])
                    * np.exp(-np.abs(pH - 7) / params[10]))
    # Final growth rate calculation
    r = combined_effect * decay_effect
    return r


def equation(b, s, temp, pH, params):
    # Monod equation for substrate concentration (saturation effect)
    # Growth rate influenced by population density and substrate concentration
    growth_rate = params[0] * (b / (params[1] + b)) * (s / (params[2] + s) + params[8] * (params[9] - s))
    # Modified Monod kinetics for both b and s
    # Temperature effect modeled with Arrhenius equation (exponential temperature dependence)
    growth_rate *= np.exp(params[3] * (temp - 20) + params[9] * (temp - 37))
    # Temperature effect with an additional term
    # pH effect with a sigmoidal response around neutral pH (pH 7)
    pH_effect = 1 / (1 + np.exp(params[4] * (7 - pH) + params[5] * (pH - 7)))
    # Steepness can be tuned with additional parameters
    growth_rate *= pH_effect
    # Fitness penalty for extreme conditions
    fitness_penalty = np.exp(-(params[6] * (temp - 37) ** 2 + params[7] * (pH - 7) ** 2))
    growth_rate *= fitness_penalty
    return growth_rate


def equation(x, v, params):
    """
    Computes the acceleration 'a' of a damped nonlinear oscillator system with advanced adaptive damping influenced by both position and velocity.

    Parameters:
        x (np.ndarray): Current position of the system.
        v (np.ndarray): Current velocity of the system.
        params (np.ndarray): Array containing model parameters:
            - params[0] = linear damping coefficient,
            - params[1] = nonlinear damping coefficient,
            - params[2] = exponent for nonlinearity,
            - params[3] = coefficient for velocity square damping term,
            - params[4] = feedback coefficient influenced by position,
            - params[5] = time-dependent external influence frequency,
            - params[6] = amplitude of external influence,
            - params[7] = time-dependent external influence frequency,
            - params[8] = constant term for driving force,
            - params[9] = additional external influence term based on environmental conditions.

    Returns:
        np.ndarray: The computed acceleration 'a' of the system.
    """

    # Caching variables for efficiency
    abs_v = np.abs(v)  # Absolute value of velocity
    v_squared = v ** 2  # Square of velocity
    sin_term = params[6] * np.sin(params[7] * x)  # Time-dependent external driving influence

    # Define damping components using cached variables
    linear_damping = params[0]  # Linear damping
    nonlinear_damping = params[1] * (abs_v ** params[2])  # Nonlinear damping
    velocity_square_damping = params[3] * v_squared  # Damping due to velocity squared
    position_dependent_damping = params[4] * x  # Damping influenced by position

    # Adaptive damping influenced by position, velocity, and external forces
    adaptive_damping = linear_damping + nonlinear_damping + velocity_square_damping + position_dependent_damping

    # Define external driving force including additional terms
    driving_force = sin_term + params[8] + params[9]  # Include constant and time-dependent influences

    # Calculate acceleration considering all contributing factors
    a = - (adaptive_damping * v) + driving_force

    return a


import numpy as np

def equation(b, s, temp, pH, params):
    # Substrate effect with saturation
    substrate_effect = s / (s + params[0])

    # Polynomial interaction for temperature and pH
    temp_pH_interaction = (temp ** params[1]) * (pH ** params[2])

    # Combined dynamic coefficient considering interactions between substrate, population density, and temperature-pH
    dynamic_coefficient = params[3] * substrate_effect * b * temp_pH_interaction

    # Interaction damping based on deviations from optimal temperature and pH
    interaction_damping = (params[4] * dynamic_coefficient) / \
                          (1 + np.abs(temp - params[5]) + np.abs(pH - params[6])) ** params[7]

    # Combined effect considering previous interaction damping and additional parameters
    combined_effect = params[8] * interaction_damping

    # Exponential decay based on distance from optimal temperature and pH
    decay_effect = np.exp(-np.abs(temp - 37) / params[9]) * \
                   np.exp(-np.abs(pH - 7) / params[10])

    # Final growth rate calculation
    r = combined_effect * decay_effect

    return r

import numpy as np

def equation(x, v, params):
    a = params[0] * x + params[1] * v - params[2] * np.sin(params[3] * x) * np.cos(params[4] * x) + params[5] * x * v + params[6] * np.exp(np.abs(x * v)) + params[7] * np.sin(v**3) + params[8] * np.exp(x * v) + params[9] * np.tanh(params[10] * v) + params[11] * np.exp(x * v)
    return a

import numpy as np

def equation(x, v, params):
    a = params[0] * np.tanh(x) + params[1] * np.arcsinh(v) + params[2] * np.log(np.abs(x)) + params[3] * x**3 + params[4] * v**3 + params[5] * x * v
    return a

problems = ["oscillator1", "oscillator2", "bactgrow", "stressstrain"]
mse_data_lst = []
c_fitness_data_lst = []
for problem_name in problems:
    mse_data = []
    c_fitness_data = []
    file_path = os.path.join('./results_1_31_17_27/llm_sr/gpt-3.5-turbo', problem_name, "pops_best")
    sample_file = os.listdir(file_path)
    sorted_files = sorted(
        sample_file,
        key=lambda x: int(x.split('=')[1].split('.')[0])  # 提取文件名中的 fe 值并转换为整数
    )

    for sample in sorted_files:
        file_path = os.path.join('./results_1_31_17_27/llm_sr/gpt-3.5-turbo', problem_name, "pops_best", sample)
        with (open(file_path, 'r') as f):
            equ = json.load(f)
            mse = equ["mse"]
            c = equ["complex"]
            c_fitness = 0.001 * np.exp(c/50)

            mse_data.append(mse)
            c_fitness_data.append(c_fitness)

    mse_data_lst.append(mse_data)
    c_fitness_data_lst.append(c_fitness_data)

for name, mse, c_fitness in zip(problems, mse_data_lst, c_fitness_data_lst):
    plt.plot(np.arange(len(mse)), mse, label=f"{name}_mse", linestyle="-")
    plt.plot(np.arange(len(c_fitness)), c_fitness, label=f"{name}_reg", linestyle='--')
    # 图例
    plt.legend()
    plt.savefig(os.path.join('./results_1_31_17_27/llm_sr/gpt-3.5-turbo', name, "mse_reg.png"), dpi=800)
    plt.show()


import numpy as np

def equation(x, v, params):
    linear_restoring = -params[0] * x
    nonlinear_restoring = -(
        params[3] * x**3
        - params[4] * x**5
        + params[5] * np.sin(params[6] * x + params[7])
    )
    damping_term = -params[1] * v**3 - params[2] * x * v
    driving_force = params[8] * np.sin(params[9] * x + params[10])
    a = linear_restoring + nonlinear_restoring + damping_term + driving_force
    return a


import numpy as np

def equation(x, v, params):
    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10 = params
    s = np.sin
    lr = -c0*x
    nr = -c3*x**3 + c4*x**5 - c5*s(c6*x+c7)
    dm = -c1*v**3 - c2*x*v
    df =  c8*s(c9*x+c10)
    return lr + nr + dm + df


