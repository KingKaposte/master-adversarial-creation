from advattack.evolution.mutation import mutate
from advattack.evolution.fitness import get_fitness
from advattack.evolution.selection import crossover
from advattack.evolution.selection import select_best
import numpy as np


# return the best generated attack
def generate_attack(input_file, x_orig, target, target_label, sess, input_node, output_node,
                    max_query, population_size, children_size, data, eps_limit=256):
    parents = [mutate(x_orig, eps_limit, 'little') for _ in range(population_size)]

    parents_fitness = []
    for x in parents:
        fitness, _ = get_fitness(sess, x, target, input_node, output_node)
        parents_fitness.append(fitness)

    top_attack, top_fitness = get_best_attack(parents, parents_fitness)

    for idx in range(int(max_query / children_size)):
        children_selected = crossover(parents, children_size)
        children_mutated = [mutate(child, eps_limit, 'big') for child in children_selected]
        children_fitness = []
        for i in range(len(children_mutated)):
            child = children_mutated[i]
            child_fitness, is_top_prediction = get_fitness(sess, child, target, input_node, output_node)
            children_fitness.append(child_fitness)
#### if negation is active
            if is_top_prediction:
                data.append([idx * children_size + i, child_fitness, is_top_prediction, target_label, 1, input_file])
            else:
                data.append([idx * children_size + i, child_fitness + 1, is_top_prediction, target_label, 1, input_file])
            if child_fitness > top_fitness:
                top_fitness = child_fitness
                top_attack = child
        parents, parents_fitness = select_best(parents, children_mutated, parents_fitness, children_fitness)
    return top_attack, top_fitness


def get_best_attack(parents, parents_fitness):
    parents_ranks = list(reversed(np.argsort(parents_fitness)))
    return parents[parents_ranks[0]], parents_fitness[parents_ranks[0]]