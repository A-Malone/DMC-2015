import numpy as np
import csv

#--------------------------------------------------------------------
#----GENETIC ALGORITHM IMPLEMENTATION

import random
import time

#----CONSTAMTS
#Define the population size
pop_size  = 60; trait_number = 10;
elitism = True

max_fitness = 100

#Minimum and maximum values for the traits
min_val = 0; max_val = 10

#The mutation rate, in percent
mutation_rate = 3

def create_individual(size, min_val, max_val):
    """
    create an individual with the number of traits size, with each trait in
    the range specified by min_val and max_val
    """
    return np.array([random_trait(min_val, max_val) for x in range(size)])

def fitness(individual):
    """

    A simple fitness function that simply sums up the values of the traits,
    producing a value to map that individual's chance of reproduction.
    """
    global 
    return sum(individual)


def mutate_population(population, min_val, max_val):
    """
    A mutation funtion that mutates the individuals.
    """
    global mutation_rate

    for individual in population:
        for position,trait in enumerate(individual):
            if random.randint(0,100) < mutation_rate:
                individual[position] = random_trait(min_val, max_val)


def random_trait(min_val, max_val):
    """
    creates random traits for the individuals
    """
    return random.randint(min_val,max_val)

def reproduce(population):
    global pop_size, elitism

    scores = [[fitness(x),x] for x in population]
    scores.sort()

    if elitism:
        scores[-1][0]*= 2

    total = 0
    probabilities = scores
    for i in range(len(scores)):
        total += scores[i][0]
        probabilities[i][0] = total

    new_pop = []
    
    while len(new_pop)<pop_size:
        p1 = roulette(probabilities,total)
        p2 = roulette(probabilities,total)
        new_pop.append(cross_over((p1,p2)))
    return new_pop

def cross_over(parents):
    global trait_number
    index = random.randint(int(trait_number/5),int(4*trait_number/5))
    swap = bool(random.randint(0,1))
    #print(parents)
    return np.concatenate((parents[int(swap)])[:index] + (parents[int(not swap)])[index:])

def roulette(probabilities,total):
    max_val = probabilities[-1][0]
    
    choice = random.randint(0,total-1)
    for obj in probabilities:
        if choice < obj[0]:
            return obj[1]

def check_pop(population):
    global max_fitness
    for i in population:
        if fitness(i) == max_fitness:
            return i
    return None

def run_ga():
    global pop_size, trait_number, elitism

    runs = 100
    gen_max = 20000
    data = [[None]*runs for x in range(2)]
    for run in range(runs):

        #get the start time
        start_time = time.time()

        #Create an initial population
        population = [create_individual(trait_number, min_val, max_val) for x in range(trait_number)]

        for gen_count in range(gen_max):
            x = check_pop(population)
            if x:
                #print("Solution found at generation: {}, individual: {}".format(i,gen_count+1))
                data[0][run] = time.time()-start_time
                data[1][run] = gen_count+1            
                break
            new_population = reproduce(population)
            mutate_population(new_population, min_val, max_val)
            population = new_population
        else:
            print("The solution was not found")
            data[0][run] = time.time()-start_time
            data[1][run] = gen_max

    total_time = sum(data[0])
    average_time = total_time/len(data[0])

    total_generations = sum(data[1])
    average_gen = total_generations/len(data[1])

    print("Total time: {}s, Total number of generations: {}, Average Time: {}, Average number of generations: {}".format(total_time,total_generations,average_time,average_gen))

#--------------------------------------------------------------------
#----DATA MANIPULATION

file_name  = "../Data/SEM_DAILY_BUILD.csv"

dic = {}

def get_conversion_ratio(row):
    try:
        return int(row["APPLICATIONS"])/float(row["CLICKS"])        
    except:
        return 0.0

#----ACQUIRE KEYWORDS
line_count = 0
click_line_count = 0
kw_counter = 0

input_file = csv.DictReader(open(file_name, "r"))
for row in input_file:    
    if(line_count == 0): #Skip first row
        line_count += 1
        continue

    #Count lines
    line_count += 1

    if(int(row["CLICKS"]) != 0):
        click_line_count += 1
    else:
        continue            #Skip if no clicks
    
    #Count keywords
    txt = row["KEYWD_TXT"]
    kws = [int(x[2:]) for x in txt.split("+") if x != ""]
    
    cvr = get_conversion_ratio(row)

    for i in kws:        
        try:
            a = dic[i]            
        except:
            dic[i] = kw_counter
            kw_counter += 1

print("Done pre-test: {} kws , {} clickthroughs , {} lines".format(kw_counter , click_line_count, line_count))