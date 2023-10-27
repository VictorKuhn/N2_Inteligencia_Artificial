
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(best_chromosome_ga, best_fitness_ga, best_position_pso, best_fitness_pso):
    algorithms = ['GA', 'PSO']
    investment_proportions = [best_chromosome_ga, best_position_pso]
    fitness_values = [best_fitness_ga, best_fitness_pso]

    bar_width = 0.35
    index = np.arange(len(algorithms))

    fig, ax1 = plt.subplots(figsize=(10, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Algoritmo')
    ax1.set_ylabel('Proporção de Investimento', color=color)
    bars1 = ax1.bar(index, investment_proportions, bar_width, label='Proporção de Investimento', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Valor de Aptidão', color=color)
    bars2 = ax2.bar(index + bar_width, fitness_values, bar_width, label='Valor de Aptidão', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(algorithms)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()

    plt.title('Comparação entre GA e PSO')
    plt.show()

