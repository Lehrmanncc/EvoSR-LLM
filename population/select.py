import random


def best_select(pop, m):
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    parents = random.choices(pop, weights=probs, k=m)
    return parents


def tournament_select(population, m):
    tournament_size = 2
    parents = []
    while len(parents) < m:
        tournament = random.sample(population, tournament_size)
        tournament_fitness = [fit['objective'] for fit in tournament]
        winner = tournament[tournament_fitness.index(min(tournament_fitness))]
        parents.append(winner)
    return parents