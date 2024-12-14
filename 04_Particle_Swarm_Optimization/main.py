import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import random


class Customer:
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        demand: float,
        ready_time: float,
        due_date: float,
        service_time: float,
    ):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time


class Particle:
    def __init__(self, num_customers: int, vehicle_capacity: float):
        self.position = np.random.permutation(num_customers)
        self.velocity = np.zeros(num_customers)
        self.best_position = self.position.copy()
        self.best_fitness = float("inf")
        self.current_fitness = float("inf")
        self.vehicle_capacity = vehicle_capacity


class PSO_VRP:
    def __init__(
        self,
        customers: List[Customer],
        num_particles: int,
        vehicle_capacity: float,
        w: float = 0.729,
        c1: float = 1.49445,
        c2: float = 1.49445,
    ):
        self.customers = customers
        self.num_customers = len(customers)
        self.num_particles = num_particles
        self.vehicle_capacity = vehicle_capacity

        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter

        self.particles = [
            Particle(self.num_customers, vehicle_capacity) for _ in range(num_particles)
        ]

        self.global_best_position = self.particles[0].position.copy()
        self.global_best_fitness = float("inf")

        for particle in self.particles:
            fitness, is_feasible = self.calculate_route_fitness(particle.position)
            if is_feasible:
                particle.current_fitness = fitness
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

        self.fitness_history = []

    def calculate_distance(self, customer1: Customer, customer2: Customer) -> float:
        return np.sqrt(
            (customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2
        )

    def calculate_route_fitness(
        self, route: List[int], verbose: bool = False
    ) -> Tuple[float, bool]:
        total_distance = 0
        current_load = 0
        current_time = 0
        depot = self.customers[0]

        prev_customer = depot
        is_feasible = True
        constraint_violations = []

        for customer_idx in route:
            current_customer = self.customers[customer_idx]

            distance = self.calculate_distance(prev_customer, current_customer)
            total_distance += distance

            travel_time = distance  # Assuming unit speed
            current_time += travel_time

            TIME_WINDOW_TOLERANCE = 10  # Allow small violations
            if current_time > current_customer.due_date + TIME_WINDOW_TOLERANCE:
                is_feasible = False
                constraint_violations.append(
                    f"Time window violated for customer {customer_idx}"
                )

            if current_time < current_customer.ready_time:
                current_time = current_customer.ready_time

            CAPACITY_TOLERANCE = 0.1  # Allow 10% overload
            current_load += current_customer.demand
            if current_load > self.vehicle_capacity * (1 + CAPACITY_TOLERANCE):
                is_feasible = False
                constraint_violations.append(
                    f"Capacity exceeded: {current_load} > {self.vehicle_capacity}"
                )

            current_time += current_customer.service_time
            prev_customer = current_customer

        total_distance += self.calculate_distance(prev_customer, depot)

        if random.random() < 0.01 and verbose:  # 1% chance to print debug info
            print(f"\nDebug Info:")
            print(f"Route: {route}")
            print(f"Total distance: {total_distance:.2f}")
            print(f"Final load: {current_load:.2f}")
            print(f"Final time: {current_time:.2f}")
            print(f"Is feasible: {is_feasible}")
            if not is_feasible:
                print("Violations:", constraint_violations)
            print("-" * 50)

        return total_distance, is_feasible

    def update_particle_velocity(self, particle: Particle):
        r1, r2 = random.random(), random.random()

        new_velocity = np.zeros_like(particle.velocity)

        for i in range(len(particle.position)):
            if particle.position[i] != particle.best_position[i]:
                new_velocity[i] += self.c1 * r1

            if particle.position[i] != self.global_best_position[i]:
                new_velocity[i] += self.c2 * r2

        particle.velocity = self.w * particle.velocity + new_velocity

    def update_particle_position(self, particle: Particle):
        new_position = particle.position.copy()

        for i in range(len(particle.velocity)):
            if random.random() < abs(particle.velocity[i]):
                j = random.randint(0, len(particle.position) - 1)
                new_position[i], new_position[j] = new_position[j], new_position[i]

        particle.position = new_position

    def optimize(self, max_iterations: int):
        for iteration in range(max_iterations):
            best_iteration_fitness = float("inf")

            for particle in self.particles:
                fitness, is_feasible = self.calculate_route_fitness(particle.position)

                if not is_feasible:
                    fitness = fitness * 1.5  # Penalty factor

                particle.current_fitness = fitness

                if particle.current_fitness < particle.best_fitness:
                    particle.best_fitness = particle.current_fitness
                    particle.best_position = particle.position.copy()

                    if particle.best_fitness < self.global_best_fitness:
                        self.global_best_fitness = particle.best_fitness
                        self.global_best_position = particle.best_position.copy()

                best_iteration_fitness = min(best_iteration_fitness, fitness)

            for particle in self.particles:
                self.update_particle_velocity(particle)
                self.update_particle_position(particle)

            self.fitness_history.append(self.global_best_fitness)

            if iteration % 10 == 0:
                print(
                    f"Iteration: {iteration} Best fitness: {self.global_best_fitness}"
                )

    def plot_convergence(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title("PSO Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.show()

    def plot_best_route(self):
        plt.figure(figsize=(10, 10))

        # Plot customers
        x_coords = [customer.x for customer in self.customers]
        y_coords = [customer.y for customer in self.customers]
        plt.scatter(x_coords, y_coords, c="blue", label="Customers")

        # Plot depot
        plt.scatter(
            self.customers[0].x,
            self.customers[0].y,
            c="red",
            marker="s",
            s=100,
            label="Depot",
        )

        # Plot route
        if self.global_best_position is not None:
            route = self.global_best_position
            for i in range(len(route)):
                start = self.customers[route[i]]
                if i < len(route) - 1:
                    end = self.customers[route[i + 1]]
                else:
                    end = self.customers[0]  # Return to depot
                plt.plot([start.x, end.x], [start.y, end.y], "g--")

        plt.title("Best Route Found")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()


# Load and prepare data
def load_solomon_data(file_path: str) -> List[Customer]:
    df = pd.read_csv(file_path)
    customers = []
    for _, row in df.iterrows():
        customer = Customer(
            id=int(row["CUST NO."]),
            x=float(row["XCOORD."]),
            y=float(row["YCOORD."]),
            demand=float(row["DEMAND"]),
            ready_time=float(row["READY TIME"]),
            due_date=float(row["DUE DATE"]),
            service_time=float(row["SERVICE TIME"]),
        )
        customers.append(customer)
    return customers


if __name__ == "__main__":
    NUM_PARTICLES = 50
    MAX_ITERATIONS = 100
    VEHICLE_CAPACITY = 200.0

    customers = load_solomon_data("/home/chukhran/datasets/solomon_dataset/C1/C101.csv")

    pso = PSO_VRP(customers, NUM_PARTICLES, VEHICLE_CAPACITY)
    pso.optimize(MAX_ITERATIONS)

    pso.plot_convergence()
    pso.plot_best_route()

    print(f"Best fitness found: {pso.global_best_fitness}")
    print(f"Best route found: {pso.global_best_position}")
