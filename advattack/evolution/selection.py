import numpy as np

header_len = 44


def crossover(population, size):
    return [select_random(
        population[np.random.choice(len(population))],
        population[np.random.choice(len(population))])
        for _ in range(size)]


def select_best(parents, children_mutated, parents_fitness, children_fitness):
    sorted_parents_fitness = list(reversed(np.argsort(parents_fitness)))
    sorted_children_fitness = list(reversed(np.argsort(children_fitness)))
    parent_counter = 0
    children_counter = 0
    best = []
    best_fitness = []
    for i in range(0, len(parents)):
        if children_counter >= len(children_mutated):
            best.append(parents[sorted_parents_fitness[parent_counter]])
            best_fitness.append(parents_fitness[sorted_parents_fitness[parent_counter]])
            parent_counter += 1
        else:
            if parents_fitness[sorted_parents_fitness[parent_counter]] < children_fitness[sorted_children_fitness[children_counter]]:
                best.append(children_mutated[sorted_children_fitness[children_counter]])
                best_fitness.append(children_fitness[sorted_children_fitness[children_counter]])
                children_counter += 1
            else:
                best.append(parents[sorted_parents_fitness[parent_counter]])
                best_fitness.append(parents_fitness[sorted_parents_fitness[parent_counter]])
                parent_counter += 1
    return best, best_fitness


def select_random(x1, x2):
    ba1 = bytearray(x1)
    ba2 = bytearray(x2)
    step = 2
    for i in range(header_len, len(x1), step):
        if np.random.random() < 0.5:
            ba2[i] = ba1[i]
    return bytes(ba2)