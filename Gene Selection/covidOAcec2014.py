# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 10:47:58 2023

@author: biswa
"""

import numpy as np
import pandas as pd
import random
import time
from tabulate import tabulate
import csv
import statistics
from sklearn import metrics

# Constants
POPULATION_SIZE = 24#24#24
NUM_DIMENSIONS = 60  # Number of dimensions in the solution
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.01
NUM_GENERATIONS = 100#5000
NUM_TESTS = 20#20
LOWER = 0#0
UPPER = 999#9
BIG_VAL = NUM_DIMENSIONS * (UPPER ** 2)

# Fitness functions
##cec2014_begin
def knn_10_fold(x):
    #x = np.array(x)
    #x = x.astype(int)  # Check if this integer conversion is appropriate for your data
    data = pd.read_csv('selected_genes_normalized.csv')
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.fillna(0)
    target = data['class']
    data.drop(['class'], axis=1, inplace=True)
    data = data.T
    data = data.to_numpy()
    data_solution = []

    for j in range(len(x)):
        ss = data[int(x[j])]
        data_solution.append(ss)

    data_solution = np.array(data_solution)
    data = pd.DataFrame(data_solution)
    data = data.T
    X = np.array(data)
    y = np.array(target)

    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    Accuracy = []

    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        Accuracy.append(metrics.accuracy_score(y_test, y_pred) * 100)

    k_fold_accuracy_score = np.mean(Accuracy)
    return k_fold_accuracy_score

def knn(x):
        import numpy as np
        x = np.array(x)
        x = x.astype(int)  # Check if this integer conversion is appropriate for your data
        data = pd.read_csv('selected_genes_normalized.csv')
        data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
        data = data.fillna(0)  # Replace NaN values with zeros
        target = data['class']
        data.drop(['class'], axis=1, inplace=True)
        data = data.T
        data = data.to_numpy()
        data_solution = []
        for j in range(len(x)):
            ss = data[x[j]]
            data_solution.append(ss)
        data_solution = np.array(data_solution)
        data = pd.DataFrame(data_solution)
        data = data.T
        X = data
        y = target
        X = np.array(X)
        y = np.array(y)
        import numpy as np
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        loo.get_n_splits(X)

        Accuracy = []
        for train_index, test_index in loo.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #print(X_train, X_test, y_train, y_test)

            #Import KNN model
            from sklearn.neighbors import KNeighborsClassifier  

            #Create a svm Classifier
            clf = KNeighborsClassifier( n_neighbors=5,metric='minkowski', p=4)


            #Train the model using the tra ning sets
            clf.fit(X_train, y_train)

            #Predict the response for test dataset
            y_pred = clf.predict(X_test)
            # Model Accuracy: how often is the classifier correct?
            #print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%")
            Accuracy.append(metrics.accuracy_score(y_test, y_pred)*100)
        LeaveOneOut_accuracy_score = np.mean(Accuracy)
        #print (LeaveOneOut_accuracy_score)
        return LeaveOneOut_accuracy_score
    #kfold maximise output

def sphere_function(x):
    x = np.array(x)
    result = np.sum(x**2)
    return 1/result

def zakharov_function(x):
    x = np.array(x)
    n = len(x)
    term1 = np.sum(x**2)
    term2 = np.sum([0.5 * (i + 1) * x[i] for i in range(n)])
    result = term1 + term2**2 + term2**4
    return 1/result


# =============================================================================
# def zakharov_function(solution):
#     term1 = sum(x**2 for x in solution)
#     term2 = sum(0.5 * x for x in solution)
#     #term2 = sum(0.5 * (i + 1) * x for i, x in enumerate(solution))
#     result = (term1 + term2**2 + term2**4)
#     return 1/result
# =============================================================================

def dixon_price_function(x):
    n = len(x)
    term1 = (x[0] - 1)**2

    sum_term = 0
    for i in range(1, n):
        term2 = (i + 1) * (2 * x[i]**2 - x[i - 1])**2
        sum_term += term2

    result = term1 + sum_term
    return 1/result

def high_conditioned_elliptic_function(x):
    n = len(x)
    result = sum((10**6)**(i / (n - 1)) * x[i]**2 for i in range(n))
    return 1/result


def bent_cigar(x):
    x = np.array(x)
    n = len(x)
    y = x[1:n]
    result = x[0]**2 + 1e6 * np.sum(y**2)
    return 1/result

def discus(x):
    x = np.array(x)
    n = len(x)
    y = x[1:n]
    result = 1e6 * x[0]**2 + np.sum(y**2)
    return 1/result

def rosenbrock(x):
    x = np.array(x)
    result = np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return 1/result

def ackley(x):
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    result = -20.0 * np.exp(-0.2 * np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.exp(1)
    return 1/result

def weierstrass(x):
    x = np.array(x)
    n = len(x)
    a, b, k_max = 0.5, 3.0, 20
    result = 0.0
    for i in range(n):
        term1 = np.sum([a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5)) for k in range(k_max)])
        result += term1
    term2 = np.sum([a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max)])
    return 1/(result - n * term2)

def griewank(x):
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2) / 4000.0
    prod2 = np.prod(np.cos(x / np.sqrt(np.arange(1, n+1))))
    result = 1.0 + sum1 - prod2
    return 1/result

def rastrigin(x):
    x = np.array(x)
    n = len(x)
    result = 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    return 1/result


# =============================================================================
# def katsuura(x):
#     x = np.array(x)
#     n = len(x)
#     product = 1.0
#     for i in range(n):
#         term = 1.0
#         for j in range(1, 33):
#             term *= np.abs(2**j * x[i] - np.ceil(2**j * x[i])) / 2**j
#         product *= (1 + (i + 1) * term) ** (n ** (10/1.2))
#     result = 10.0 / n**2 * product - 10.0 / n**2
#     print(result)
#     return 1/result
# =============================================================================


def happy_cat(x):
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(x)
    result = (np.abs(sum1 - n))**0.25 + (0.5 * sum1 + sum2) / n + 0.5
    return 1/result

def h_g_bat(x):
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(x)
    result = (np.abs(sum1**2 - sum2**2))**0.5 + (0.5 * sum1 + sum2) / n + 0.5
    return 1/result

def expanded_griewank_rosenbrock(x):
    x = np.array(x)
    n = len(x)
    result = 0.0
    for i in range(n-1):
        result += rosenbrock(x[i:i+2])
    result += griewank(x)
    return 1/result

def expanded_scaffer_F6(x):
    x = np.array(x)
    n = len(x)
    result = 0.0
    for i in range(n-1):
        result += scaffer_F6(x[i:i+2])
    result += scaffer_F6(np.array([x[-1], x[0]]))
    return 1/result

def scaffer_F6(x):
    x = np.array(x)
    n = len(x)
    result = 0.0 
    for i in range(n-1):
        result += 0.5 + (np.sin(np.sqrt(x[i]**2 + x[i+1]**2))**2 - 0.5) / (1 + 0.001 * (x[i]**2 + x[i+1]**2))
    return 1/result
##cec2014_end

# List of fitness function
#fitness_functions = [high_conditioned_elliptic, bent_cigar, discus, rosenbrock, ackley, weierstrass, griewank, rastrigin, happy_cat, h_g_bat, expanded_griewank_rosenbrock, expanded_scaffer_F6, scaffer_F6]
#fitness_functions = [high_conditioned_elliptic, bent_cigar, discus, rosenbrock, ackley, weierstrass, griewank, rastrigin, happy_cat, h_g_bat]
fitness_functions = [knn_10_fold]

#Select parents for reproduction using roulette wheel selection (Higher fitness Lower Probability)
def roulette_wheel_selection(population, fitness_function):
    total_fitness = sum(fitness_function(individual) for individual in population)
    probabilities = [(fitness_function(individual) / total_fitness) for individual in population]
    prev = 0
    prefixSum = []
    for prob in probabilities:
        prev = prev + prob
        prefixSum.append(prev)
    selected_parents = []
    while len(selected_parents) < 2:
        r = random.random()
        for i in range(len(prefixSum)):
            if(r < prefixSum[i]):
                parent = population[i]
                break
        if parent not in selected_parents:
            selected_parents.append(parent)
    return selected_parents

# Generate an initial population of solutions
def generate_population(population_size, num_dimensions):
    return [[random.randint(LOWER, UPPER) for _ in range(num_dimensions)] for _ in range(population_size)]

# FrameShifting
def frame_shifting(parent):
    shift_direction = random.choice([-1, 1])
    if shift_direction == 1:
        protein = [random.randint(LOWER, UPPER)] + parent[:-1]
    else:
        protein = parent[1:] + [random.randint(LOWER, UPPER)]
    return protein

# Crossover function (uniform crossover)
def uniform_crossover(protein1, protein2, crossover_rate):
    virus1 = []
    virus2 = []
    for ind in range(len(protein1)):
        if(random.random() < crossover_rate):
            virus1.append(protein2[ind])
            virus2.append(protein1[ind])
        else:
            virus1.append(protein1[ind])
            virus2.append(protein2[ind])
    return virus1, virus2

# Mutation function
def mutate(virus, mutation_rate):
    for i in range(len(virus)):
        if random.random() < mutation_rate:
            virus[i] = random.randint(LOWER, UPPER)
    return virus

def find_best_individual_and_fitness(population, fitness_function):
    best_individual = None
    best_fitness = float('-inf')  # Initialize to negative infinity

    for individual in population:
        current_fitness = fitness_function(individual)
        if current_fitness > best_fitness:
            best_individual = individual
            best_fitness = current_fitness

    return best_individual, best_fitness

def create_selected_sample_file(selected_samples):
    #selected_samples = np.array(selected_samples).astype(int)
    data = pd.read_csv('selected_genes.csv')
    selected_data = data.iloc[:, selected_samples]
    selected_data.to_csv('selected_samples.csv', index = False)
    


# Main genetic algorithm loop
def genetic_algorithm(fitness_function):
    
    best_individuals = []
    results = []
    
    start_time = time.time()
    for _ in range(NUM_TESTS):
        print(f'Run: {_}')
        population = generate_population(POPULATION_SIZE, NUM_DIMENSIONS)
        best_individual_gen, best_individual_gen_fitness = find_best_individual_and_fitness(population, fitness_function)
        #print("start")
        for generation in range(NUM_GENERATIONS):
            print(generation)
            new_population = []
            
            for _ in range(POPULATION_SIZE // 2):
                #print(population)
                parent1, parent2 = roulette_wheel_selection(population, fitness_function)
                protein1 = frame_shifting(parent1)
                protein2 = frame_shifting(parent2)
                virus1, virus2 = uniform_crossover(protein1, protein2, CROSSOVER_RATE)
                virus1 = mutate(virus1, MUTATION_RATE)
                virus2 = mutate(virus2, MUTATION_RATE)
                new_population.extend([virus1, virus2])
            population = new_population
            best_individual_cur_gen, best_individual_cur_gen_fitness = find_best_individual_and_fitness(population, fitness_function)
            if(best_individual_cur_gen_fitness > best_individual_gen_fitness):
                best_individual_gen = best_individual_cur_gen
                best_individual_gen_fitness = best_individual_cur_gen_fitness
            #min_solution = max(population, key=fitness_function)
            #print(f"{min_solution}: {1/fitness_function(min_solution)}")
        #print("end")
        best_individuals.append(best_individual_gen)
        df = pd.DataFrame({"Column_Name": best_individuals})
        df.to_csv("bestindividuals.csv", index=False)
        
        
    end_time = time.time()
    time_taken = end_time - start_time
    best_individuals_fitness_values = [fitness_function(individual) for individual in best_individuals]
    for ind in range(len(best_individuals)):
        print(f"{best_individuals[ind]}: {best_individuals_fitness_values[ind]}")
    min_individual = min(best_individuals, key=fitness_function)
    max_individual = max(best_individuals, key=fitness_function)
    
    genes = pd.read_csv('selected_genes_normalized.csv')
    selected_gene_names = genes.columns[max_individual]
    selected_gene_names_list = selected_gene_names.tolist()
    print(f'Selected Gene index: {max_individual}')
    print(f'Selected Gene names: {selected_gene_names_list}')
    
    best = fitness_function(min_individual)
    worst = fitness_function(max_individual)
    mean = statistics.mean(best_individuals_fitness_values)
    standard_deviation = statistics.stdev(best_individuals_fitness_values)
    median = statistics.median(best_individuals_fitness_values)
    results.append([fitness_function.__name__, best, worst, mean, median, standard_deviation, time_taken])
    create_selected_sample_file(max_individual)
    return results

#Printing Constants
print(f"Population size: {POPULATION_SIZE}")
print(f"Number Dimension: {NUM_DIMENSIONS}")
print(f"Mutation rate: {MUTATION_RATE}")
print(f"Number of Generation: {NUM_GENERATIONS}")
print(f"Number of Test Runs: {NUM_TESTS}")
print()


# Run the genetic algorithm for each fitness function
all_results = []
for fitness_function in fitness_functions:
    print(f"Optimizing for {fitness_function.__name__}:")
    results = genetic_algorithm(fitness_function)
    all_results.extend(results)
    print()
    
# Print the results in tabular format
headers = ["Fitness Function", "Min Fitness", "Max Fitness", "Mean", "Median", "Standard Deviation", "Time Taken (s)"]
table = tabulate(all_results, headers=headers, tablefmt="grid")
print(table)


# writing to csv file 
with open("gentecAlgoOutput.csv", 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(headers) 
        
    # writing the data rows 
    csvwriter.writerows(all_results)