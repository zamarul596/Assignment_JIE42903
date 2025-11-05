import streamlit as st
import csv
from pathlib import Path
import random
import pandas as pd

#######################################
# Function to read CSV
#######################################
def read_csv_to_dict(file_path):
    p = Path(file_path)
    if not p.exists():
        st.error(f"CSV file not found at: {p}")
        return {}

    program_ratings = {}
    with p.open(mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        try:
            next(reader)  # skip header
        except StopIteration:
            return {}

        for row in reader:
            if not row:
                continue
            program = row[0].strip()
            try:
                ratings = [float(x) for x in row[1:] if x != ""]
            except ValueError as e:
                st.error(f"Invalid numeric value in CSV for program '{program}': {e}")
                return {}
            program_ratings[program] = ratings

    return program_ratings


#######################################
# Genetic Algorithm Functions
#######################################
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating


def crossover(schedule1, schedule2):
    if len(schedule1) < 3:
        return schedule1.copy(), schedule2.copy()
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2


def mutate(schedule, all_programs):
    if not schedule:
        return schedule
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule


def genetic_algorithm(ratings, all_programs, generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):

    num_time_slots = 18  # From 6:00 to 23:00
    # Initialize population (programs can repeat)
    population = []
    for _ in range(population_size):
        schedule = [random.choice(all_programs) for _ in range(num_time_slots)]
        population.append(schedule)

    for _ in range(generations):
        # Evaluate fitness
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_population = population[:elitism_size]  # Elitism

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)

            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    return population[0]


#######################################
# Display Schedule Function
#######################################
def display_schedule(schedule, ratings, title, co_r, mut_r):
    all_time_slots = list(range(6, 24))  # 6:00 to 23:00

    total_rating = 0
    results = []
    for time_slot, program in enumerate(schedule):
        hour = all_time_slots[time_slot]
        rating = ratings[program][time_slot]
        total_rating += rating
        results.append({
            "Time Slot": f"{hour:02d}:00",
            "Program": program,
            "Rating": rating
        })

    df = pd.DataFrame(results)
    st.subheader(title)
    st.write(f"**Crossover Rate:** {co_r} | **Mutation Rate:** {mut_r}")
    st.dataframe(df, use_container_width=True)
    st.success(f"Total Ratings: {total_rating:.2f}")


#######################################
# Streamlit Interface
#######################################
st.title("Optimal TV Program Scheduler (Genetic Algorithm)")
st.write("This app finds the best TV program schedule using a Genetic Algorithm.")

st.markdown("### Step 1: Set Parameters for Each Trial")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Default Parameters (Trial 0)**")
    default_CO_R = 0.8
    default_MUT_R = 0.2
    st.write(f"Crossover Rate: **{default_CO_R}**")
    st.write(f"Mutation Rate: **{default_MUT_R}**")

with col2:
    st.markdown("**Parameter Ranges**")
    st.write("- Crossover Rate (CO_R): 0.0 – 0.95")
    st.write("- Mutation Rate (MUT_R): 0.01 – 0.05")

# Sliders for 3 trials
st.markdown("---")
st.markdown("1) Trial Parameters")

trial_params = []
for i in range(1, 4):
    st.subheader(f"Trial {i}")
    co_r = st.slider(f"Trial {i} - Crossover Rate", 0.0, 0.95, 0.8, 0.01, key=f"co_r_{i}")
    mut_r = st.slider(f"Trial {i} - Mutation Rate", 0.01, 0.05, 0.02, 0.01, key=f"mut_r_{i}")
    trial_params.append((co_r, mut_r))

st.markdown("---")
st.markdown("### Step 2: Upload CSV File")

uploaded_file = st.file_uploader("Upload your program_ratings.csv file", type=["csv"])

if uploaded_file:
    temp_path = Path("program_ratings.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ratings = read_csv_to_dict(temp_path)

    if not ratings:
        st.warning("No valid data found in CSV.")
        st.stop()

    all_programs = list(ratings.keys())

    st.markdown("### Step 3: Run the Genetic Algorithm")
    if st.button("Run All Trials"):
        st.header("Final Optimal Schedules")

        # Default Run
        best_default = genetic_algorithm(ratings, all_programs,
                                         crossover_rate=default_CO_R,
                                         mutation_rate=default_MUT_R)
        display_schedule(best_default, ratings, "Default Run Results", default_CO_R, default_MUT_R)

        # User Trials
        for i, (co_r, mut_r) in enumerate(trial_params, start=1):
            best_trial = genetic_algorithm(ratings, all_programs,
                                           crossover_rate=co_r,
                                           mutation_rate=mut_r)
            display_schedule(best_trial, ratings, f"Trial {i} Results", co_r, mut_r)

else:
    st.info("Please set the parameters above and then upload your CSV file to continue.")
