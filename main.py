import data_processing as dp
import genetic_algorithm as ga
import pso
from visualization import plot_comparison


def main():
    data = dp.load_data("apple_data.csv")
    daily_returns = dp.calculate_daily_returns(data)['Daily_Return']

    best_chromosome_ga, best_fitness_ga = ga.genetic_algorithm(daily_returns)
    print(
        f"GA - Melhor proporção de investimento: {best_chromosome_ga * 100:.2f}%, Valor de aptidão: {best_fitness_ga:.4f}")

    swarm = pso.Swarm(num_particles=100, dimension=1,
                      fitness_function=lambda x: ga.fitness_function(x[0], daily_returns))
    best_position, best_fitness_pso = swarm.optimize()
    print(
        f"PSO - Melhor proporção de investimento: {best_position[0] * 100:.2f}%, Valor de aptidão: {best_fitness_pso:.4f}")

    return best_chromosome_ga, best_fitness_ga, best_position[0], best_fitness_pso


if __name__ == "__main__":
    best_chromosome_ga, best_fitness_ga, best_position_pso, best_fitness_pso = main()
    plot_comparison(best_chromosome_ga, best_fitness_ga, best_position_pso, best_fitness_pso)

