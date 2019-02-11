#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import operator
import pandas as pd


# In[2]:


class City:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def distanceBetweenCity(self,city):
        xDistance = abs(self.x - city.x)
        yDistance = abs(self.y - city.y)
        distance = np.sqrt((xDistance ** 2) + (yDistance ** 2))
        return distance
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


# In[3]:


def fitness(route):
    pathDistance = 0
    for i in range(len(route)):
        city1 = route[i]
        city2 = None
        if i+1 == len(route):
            city2 = route[0]
        else:
            city2 = route[i+1]
        pathDistance += city2.distanceBetweenCity(city1)
#     print(pathDistance)
    fitValue = 1/float(pathDistance)
    return fitValue
#     return 0


# In[4]:


def sortPopulation(population):
    fitnessScores = {}
    for i in range(len(population)):
        fitnessScores[i] = fitness(population[i])
#     print("before sorting {}".format(fitnessScores))
    fitnessScores = sorted(fitnessScores.items(),key = lambda x: x[1], reverse = True)
#     print("after sorting {}".format(fitnessScores))
    return fitnessScores


# In[5]:


# for initializing first generation routes
def firstGenRoute(Allcity):
    temp = random.sample(Allcity,len(Allcity))
    return temp


# In[6]:


def selection(sortedPopulation, freePass):
    selectedIndex = []
    tempS = np.array(sortedPopulation)
    for i in range(0, freePass):                        #superior candidates gets free pass
        selectedIndex.append(sortedPopulation[i][0])
    remaining = len(sortedPopulation) - freePass
    for i in range(len(sortedPopulation)-1):
        tempS[i+1][1] += tempS[i][1]

    totalSum = sortedPopulation[len(sortedPopulation)-1][1]
#     print(" Total Sum in selection ",totalSum)
#     print(" Remaining ",remaining)
    for i in range(len(sortedPopulation)):
        tempS[i][1] = 100*(tempS[i][1]/float(totalSum))

    for i in range(int(remaining)):
        pick = 100 * random.random()
        for j in range(len(sortedPopulation)):
            if pick <= tempS[j][1]:
                selectedIndex.append(sortedPopulation[j][0])
                break
    return selectedIndex


# In[7]:


def findParents(population, selectedIndex):
    parentsToMate = []
    for i in range(len(selectedIndex)):
        index = selectedIndex[i]
        parentsToMate.append(population[index])
    return parentsToMate


# In[8]:


#order Crossover

def crossover(parent1, parent2):
    child = []
    
    rand1 = round(random.random() * len(parent1))
    rand2 = round(random.random() * len(parent2))
    
    firstDivider = min(rand1, rand2)
    secondDivider = max(rand1, rand2)

    for i in range(firstDivider, secondDivider):
        child.append(parent1[i])
    for i in parent2:
        if i not in child:
            child.append(i)
#     print (child)
    return child


# In[9]:


def mating(parentsToMate, freePass):
    children = []
    parentsToMate = random.sample(parentsToMate, len(parentsToMate))

    for i in range(freePass):
        children.append(parentsToMate[i])

    remaining = len(parentsToMate) - freePass    
#     print ("remaining ",remaining," ",freePass)
    for i in range(0, remaining):
        child = crossover(parentsToMate[i], parentsToMate[len(parentsToMate)-i-1])
        children.append(child)
#     print(children)
    return children


# In[10]:


def mutation(population, mutationRate):
    mutatedChild = []
#     print(type(population))
    for i in range(len(population)):
        rand = float(random.random())
        if (random.random() < mutationRate):
            index1 = int(random.random()*len(population[i]))
            index2 = int(random.random()*len(population[i]))
            temp1 = population[i][index1]
            temp2 = population[i][index2]
            population[i][index2] = temp1
            population[i][index1] = temp2
        mutatedChild.append(population[i])
    return mutatedChild


# In[11]:


def nextGeneration(population, mutationRate, freePass):
    nextGen = []
#     print(type(mutationRate))
#     print(" length of population ",len(population))
#     print(" length of sortedpopulation ",len(sortedPopulation))
#     print(" length of selectedIndex ",len(selectedIndex))
#     print(" length of parentstomate ",len(parentsToMate))
#     print(" length of children ",len(children))
    sortedPopulation = sortPopulation(population)
    selectedIndex = selection(sortedPopulation, freePass)
    parentsToMate = findParents(population, selectedIndex)
    children = mating(parentsToMate, freePass)
    children = mutation(children, mutationRate)
    return children


# In[12]:


def geneticAlgo(population, populationSize, mutationRate, generations, freepass):
    print(" First Route Distance :- {}".format(1/sortPopulation(population)[0][1]))
    print ("First Route ",population[sortPopulation(population)[0][0]])
#     print(sortPopulation(population))
    currentBest = sortPopulation(population)[0][1]
    wait = 0
    for i in range(generations):
#         print("Generation {} Results".format(i+1))
        population = nextGeneration(population,mutationRate,freepass)

    print("Final distance: " + str(1 / sortPopulation(population)[0][1]))
    print ("Best Route ",population[sortPopulation(population)[0][0]])


# In[13]:


numberOfCities = int(input("Give number of city :- "))
populationSize = int(input("Population Size for Algorithm :- "))
mutationRate = float(input("Give Mutataion Rate :- "))
generations = int(input("Number of Generation to run the Algorithm :- "))
freePass = round(populationSize * 0.1)


# In[14]:


#generating random city Co-ordinates

Allcity = []
print ("Generating {} number of city with random Co-ordinates ".format(numberOfCities))
for i in range(int(numberOfCities)):
    city = City(x=int(random.random() * 200), y = int(random.random() * 200))
    Allcity.append(city)
# print (Allcity)


# In[15]:


#initialize First Generation population
population = []
for i in range(populationSize):
    temp = firstGenRoute(Allcity)
    population.append(temp)


# geneticFunction(allCity,populatonSize,mutation,generation,pass)
# geneticfuntion requires paramater:
# First paramater is -> co-ordinates of all city 
# second parameter is-> population size
# third paramter is -> mutaion rate
# fourth paramter is-> number of generation it has to run
# fifth parameter is-> number of best candidates to pass in every generation

# In[16]:


geneticAlgo(population,populationSize,mutationRate,generations,freePass)


# 
