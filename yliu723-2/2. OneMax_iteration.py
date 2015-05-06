import sys
import os
import time
import timeit

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.CountOnesEvaluationFunction as CountOnesEvaluationFunction
from array import array

N=200
fill = [2] * N
ranges = array('i', fill)

ef = CountOnesEvaluationFunction()
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


##tweak iteration times
rhc = RandomizedHillClimbing(hcp)
x =[]
y =[]
for i in range(10, 500, 20):
    fit = FixedIterationTrainer(rhc, i)
    fit.train()
    x.append(i)
    y.append(ef.value(rhc.getOptimal()))
print "RHC: " + str(ef.value(rhc.getOptimal()))
print x
print y


sa = SimulatedAnnealing(100, .95, hcp)
x =[]
y =[]
for i in range(10, 500, 20):
    fit = FixedIterationTrainer(sa, i)
    fit.train()
    x.append(i)
    y.append(ef.value(sa.getOptimal()))
print "SA: " + str(ef.value(sa.getOptimal()))
print x
print y

print "######### GA ##############"  
## GA perf improved as pop size increased, possibly by preserving more diversity.
## In contrast, the rates of cross-over and mutation seem to have a sweet-spot. The algo needs a certain rate to keep the diversity, 
## but increase pass the spot will not further benefit the final selection as the changes (mutation sites or cross-over points) are random and not necessarily beneficial. 
## This probably reflects the trade-off between keeping the elites and increasing the diversity.   
x =[]
y =[]
for i in range(10, 500, 20):
    ga = StandardGeneticAlgorithm(50, 20, 5, gap)
    fit = FixedIterationTrainer(ga, i)
    fit.train()
    
    x.append(i)
    y.append(ef.value(ga.getOptimal()))
print "GA: " + str(ef.value(ga.getOptimal()))
print x
print y

##tweak GA parameters
print "#########fitness vs population size##############"
x =[]
y =[]
for i in range(5, 200, 5):
    ga = StandardGeneticAlgorithm(i, i/4, 5, gap)
    fit = FixedIterationTrainer(ga, i)
    fit.train()
    
    x.append(i)
    y.append(ef.value(ga.getOptimal()))
print "GA: " + str(ef.value(ga.getOptimal()))
print x
print y

print "#########fitness vs toMate ##############"
x =[]
y =[]
for i in range(5, 200, 5):
    ga = StandardGeneticAlgorithm(195, i, 5, gap)
    fit = FixedIterationTrainer(ga, i)
    fit.train()
    
    x.append(i)
    y.append(ef.value(ga.getOptimal()))
print "GA: " + str(ef.value(ga.getOptimal()))
print x
print y

print "#########fitness vs toMutate ##############"
x =[]
y =[]
for i in range(5, 200, 5):
    ga = StandardGeneticAlgorithm(195, 45, i, gap)
    fit = FixedIterationTrainer(ga, i)
    fit.train()
    
    x.append(i)
    y.append(ef.value(ga.getOptimal()))
print "GA: " + str(ef.value(ga.getOptimal()))
print x
print y

##MIMIC
x =[]
y =[]
for i in range(10, 500, 20):
    mimic = MIMIC(50, 10, pop)
    fit = FixedIterationTrainer(mimic, i)
    fit.train()
    x.append(i)
    y.append(ef.value(mimic.getOptimal()))
print "MIMIC: " + str(ef.value(mimic.getOptimal()))
print x
print y


