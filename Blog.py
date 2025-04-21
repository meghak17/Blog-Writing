import numpy as np

import random

import matplotlib.pyplot as plt

 

class AntColony:

    def __init__(self, distance_matrix, n_ants, n_iterations,

 alpha=1, beta=2, evaporation_rate=0.5, q=100):

        self.distance_matrix = distance_matrix

        self.pheromone = np.ones(distance_matrix.shape) /

len(distance_matrix)

        self.n_ants = n_ants

        self.n_iterations = n_iterations

        self.alpha = alpha  # Pheromone importance

        self.beta = beta    # Heuristic importance

        self.evaporation_rate = evaporation_rate

        self.q = q  # Pheromone deposit factor

        self.n_cities = distance_matrix.shape[0]

        self.best_lengths = []

 

    def run(self):

        best_path = None

        best_path_length = float('inf')

       

        for iteration in range(self.n_iterations):

            all_paths = self.construct_solutions()

            self.update_pheromones(all_paths)

           

            shortest_path, shortest_path_length = min(all_paths,

 key=lambda x: x[1])

           

            if shortest_path_length < best_path_length:

                best_path = shortest_path

                best_path_length = shortest_path_length

               

            self.best_lengths.append(best_path_length)

            print(f"Iteration {iteration + 1}: Best Path Length =

 {best_path_length}")

       

        return best_path, best_path_length

 

    def construct_solutions(self):

        all_paths = []

        for _ in range(self.n_ants):

            path = self.construct_path()

            path_length = self.calculate_path_length(path)

            all_paths.append((path, path_length))

        return all_paths

   

    def construct_path(self):

        path = []

        visited = set()

        current_city = random.randint(0, self.n_cities - 1)

        path.append(current_city)

        visited.add(current_city)

       

        while len(path) < self.n_cities:

            next_city = self.select_next_city(current_city,

visited)

            path.append(next_city)

            visited.add(next_city)

            current_city = next_city

       

        path.append(path[0])  # Returning to the start city

        return path

   

    def select_next_city(self, current_city, visited):

        probabilities = []

        denominator = sum((self.pheromone[current_city][j] **

self.alpha) *

       ((1 / self.distance_matrix[current_city][j]) ** self.beta)

        for j in range(self.n_cities) if j not in visited)

       

        for j in range(self.n_cities):

            if j in visited:

                probabilities.append(0)

            else:

                numerator = (self.pheromone[current_city][j] **

 self.alpha) *

         ((1 / self.distance_matrix[current_city][j]) ** self.beta)

                probabilities.append(numerator / denominator)

       

     return random.choices(range(self.n_cities), probabilities)[0]

   

    def calculate_path_length(self, path):

        return sum(self.distance_matrix[path[i]][path[i + 1]]

         for i in range(len(path) - 1))

   

    def update_pheromones(self, all_paths):

        self.pheromone *= (1 - self.evaporation_rate)  

       

        for path, path_length in all_paths:

            for i in range(len(path) - 1):

                self.pheromone[path[i]][path[i + 1]] += self.q /

path_length

                self.pheromone[path[i + 1]][path[i]] += self.q /

path_length  # Undirected graph

 

    def plot_results(self):

        plt.plot(self.best_lengths, marker='o', linestyle='-',

color='b')

        plt.xlabel('Iteration')

        plt.ylabel('Best Path Length')

        plt.title('Convergence of Ant Colony Optimization for TSP')

        plt.grid()

        plt.show()

 
