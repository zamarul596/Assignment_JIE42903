import streamlit as st
import csv
import random
import pandas as pd

st.title("📺 Program Rating Optimizer (Lecturer’s GA Version)")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv file", type=["csv"])

if uploaded_file is not None:
    # Read CSV into dictionary
    def read_csv_to_dict(file):
        program_ratings = {}
        reader = csv.reader(file.read().decode("utf-8").splitlines())
        header = next(reader)  # skip header

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings

        return program_ratings

    program_ratings_dict = read_csv_to_dict(uploaded_file)

    ratings = program_ratings_dict
    GEN = 100
    POP = 50
    EL_S = 2

    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))

    # ---------------- SLIDERS FOR CROSSOVER & MUTATION ----------------
    st.sidebar.header("⚙️ GA Parameters")
    CO_R = st.sidebar.slider("Crossover Rate", 0.0, 0.95, 0.8, 0.01)
    MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 0.05, 0.02, 0.01)

    # ---------------- FUNCTIONS ----------------
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot]
        return total_rating

    def initialize_pop(programs, time_slots):
        if not programs:
            return [[]]
        all_schedules = []
        for i in range(len(programs)):
            for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
                all_schedules.append([programs[i]] + schedule)
        return all_schedules

    def finding_best_schedule(all_schedules):
        best_schedule = []
        max_ratings = 0
        for schedule in all_schedules:
            total_ratings = fitness_function(schedule)
            if total_ratings > max_ratings:
                max_ratings = total_ratings
                best_schedule = schedule
        return best_schedule

    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    best_schedule = finding_best_schedule(all_possible_schedules)

    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
        population = [initial_schedule]
        for _ in range(population_size - 1):
            random_schedule = initial_schedule.copy()
            random.shuffle(random_schedule)
            population.append(random_schedule)

        for generation in range(generations):
            new_population = []
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

        return population[0]

    # ---------------- RUN 3 TRIALS ----------------
    st.subheader("🎯 Running 3 Trials...")
    trial_results = []

    initial_best_schedule = finding_best_schedule(all_possible_schedules)
    rem_t_slots = len(all_time_slots) - len(initial_best_schedule)

    for trial in range(1, 4):
        genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, elitism_size=EL_S)
        final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]
        total_rating = fitness_function(final_schedule)
        trial_results.append((trial, final_schedule, total_rating))

        st.write(f"**Trial {trial} Total Rating:** {total_rating:.2f}")

    # ---------------- DISPLAY BEST RESULT IN TABLE ----------------
    best_trial = max(trial_results, key=lambda x: x[2])
    best_schedule = best_trial[1]
    best_total = best_trial[2]

    st.subheader("🏆 Best Schedule Result (Table Format)")
    df = pd.DataFrame({
        "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
        "Program": best_schedule[:len(all_time_slots)]
    })

    st.table(df)
    st.success(f"✅ Best Total Ratings: {best_total:.2f}")

else:
    st.info("👆 Please upload a CSV file to start.")
