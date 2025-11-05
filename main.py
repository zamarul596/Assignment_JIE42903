import csv
import random

####################################### LOAD DATASET #####################################################

def read_csv_to_dict(file_path):
    """
    Reads a CSV file and converts it into a dictionary format:
    { 'Program Name': [ratings from 6:00–23:00] }
    """
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header (e.g., "Program,6,7,8,...")
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # convert to float
            program_ratings[program] = ratings
    return program_ratings


# Path to your CSV file (update path if needed)
file_path = '/content/program_ratings.csv'
ratings = read_csv_to_dict(file_path)

# Confirm dataset
print("Loaded programs and ratings:")
for program, rate in ratings.items():
    print(f"{program}: {len(rate)} time slots ({rate[:3]}...)")

###################################### PARAMETERS ########################################################

GEN = 100           # Number of generations
POP = 50            # Population size
CO_R = 0.8          # Crossover rate
MUT_R = 0.2         # Mutation rate
EL_S = 2            # Elitism size

all_programs = list(ratings.keys())         # List of all programs
all_time_slots = list(range(6, 24))         # Time slots 06:00–23:00

###################################### FUNCTIONS #########################################################

def fitness_function(schedule):
    """Calculate total rating for a given schedule."""
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        # Handle case if schedule shorter than ratings length
        if time_slot < len(ratings[program]):
            total_rating += ratings[program][time_slot]
    return total_rating


def initialize_pop(programs, time_slots):
    """Generate all possible schedules (brute-force approach)."""
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules


def finding_best_schedule(all_schedules):
    """Find the best schedule by brute-force."""
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule


def crossover(schedule1, schedule2):
    """Single-point crossover between two parents."""
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2


def mutate(schedule):
    """Mutate a schedule by swapping a random program."""
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule


def evaluate_fitness(schedule):
    return fitness_function(schedule)


def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, 
                      crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
    """Run Genetic Algorithm."""
    population = [initial_schedule]

    # Create initial random population
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism — keep top individuals
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population

    # Return the best schedule found
    population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
    return population[0]

###################################### MAIN EXPERIMENT ####################################################

def run_trial(trial_num):
    print(f"\n===== TRIAL {trial_num} =====")

    # Brute force initial schedule
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules)

    # Run GA
    genetic_schedule = genetic_algorithm(initial_best_schedule, 
                                         generations=GEN, 
                                         population_size=POP, 
                                         elitism_size=EL_S)

    # Combine results
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
    final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

    print("\nFinal Optimal Schedule:")
    for time_slot, program in enumerate(final_schedule):
        print(f"Time Slot {all_time_slots[time_slot]:02d}:00 - {program}")

    total = fitness_function(final_schedule)
    print("Total Ratings:", total)
    return total

###################################### MULTIPLE TRIALS ####################################################

TRIALS = 3
best_total = 0
best_run = None

for t in range(1, TRIALS + 1):
    total = run_trial(t)
    if total > best_total:
        best_total = total
        best_run = t

print(f"\n🏆 Best Trial: {best_run} with Total Rating = {best_total}")
