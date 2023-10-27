import numpy as np


class Particle:
    def __init__(self, dimension):
        self.position = np.random.rand(dimension)
        self.velocity = np.random.rand(dimension) * 0.01
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

    def update_velocity(self, global_best_position, w=0.5, c1=1.5, c2=1.5):
        inertia = w * self.velocity
        personal_attraction = c1 * np.random.rand() * (self.best_position - self.position)
        global_attraction = c2 * np.random.rand() * (global_best_position - self.position)
        self.velocity = inertia + personal_attraction + global_attraction

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 1)

    def evaluate(self, fitness_function):
        score = fitness_function(self.position)
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position


class Swarm:
    def __init__(self, num_particles, dimension, fitness_function):
        self.particles = [Particle(dimension) for _ in range(num_particles)]
        self.global_best_position = np.random.rand(dimension)
        self.global_best_score = -float('inf')
        self.fitness_function = fitness_function

    def optimize(self, iterations=100):
        for _ in range(iterations):
            for particle in self.particles:
                particle.evaluate(self.fitness_function)
                if particle.best_score > self.global_best_score:
                    self.global_best_score = particle.best_score
                    self.global_best_position = particle.best_position
            for particle in self.particles:
                particle.update_velocity(self.global_best_position)
                particle.update_position()
        return self.global_best_position, self.global_best_score

