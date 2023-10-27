import numpy as np


def fitness_function(proportion, daily_returns):
    expected_return, risk = portfolio_performance(proportion, daily_returns)
    return expected_return / (risk + 1e-6)


def portfolio_performance(proportion, daily_returns):
    expected_return = daily_returns.mean() * proportion
    risk = daily_returns.std() * proportion
    return expected_return, risk


def select_parents(population, fitnesses):
    parents = np.random.choice(population, size=2, p=fitnesses/np.sum(fitnesses))
    return parents[0], parents[1]


def simple_crossover(parent1, parent2):
    offspring = (parent1 + parent2) / 2
    return np.clip(offspring, 0, 1)


def simple_mutate(chromosome, mutation_rate=0.02):
    if np.random.random() < mutation_rate:
        chromosome += np.random.normal(0, 0.05)
    return np.clip(chromosome, 0, 1)


def genetic_algorithm(daily_returns, population_size=100, generations=100, mutation_rate=0.02):
    population = np.random.rand(population_size)
    
    for generation in range(generations):
        fitnesses = np.array([fitness_function(chromosome, daily_returns) for chromosome in population])
        new_population = []
        for i in range(population_size):
            parent1, parent2 = select_parents(population, fitnesses)
            offspring = simple_crossover(parent1, parent2)
            new_population.append(offspring)
        new_population = [simple_mutate(chromosome, mutation_rate) for chromosome in new_population]
        population = np.array(new_population)
    best_idx = np.argmax(fitnesses)
    return population[best_idx], fitnesses[best_idx]

