import heapq

import numpy as np


def population_management(pop, size):
    pop = [individual for individual in pop if individual['objective'] is not None and
           not np.isnan(individual['objective']) and np.isreal(individual['objective'])]  # 过滤目标值为空的个体

    if size > len(pop):
        size = len(pop)
    unique_pop = []
    unique_objectives = []
    # 如果剩余个体超出规定大小，则首先选择目标值唯一的个体，确保不会出现目标值相同的个体
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual['objective'])
    # Delete the worst individual
    #pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    # 选择目标值最小的前size个个体
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])
    return pop_new
