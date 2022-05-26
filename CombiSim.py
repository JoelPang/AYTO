import pandas as pd
import random
import pickle
import os
import time
from itertools import permutations
from joblib import Parallel, delayed
import argparse

# =============================================================================
# Simulation of elimination algo logic
# De facto latest version 2022 02
# =============================================================================
# Added branches to estimate branching when picking random pair among max
# Updated pickCombination logic
# =============================================================================

def generateCombinations(N):
    print(f'Generating combinations with {N} pairs...')
    allCombinations = []
    nRange = range(1,N+1)
    for p in permutations(nRange, N):
        branch = generateBranch(p, nRange)
        allCombinations.append(branch)
    print(f'Combinations generated: {len(allCombinations):,}' )
    return allCombinations

def generateBranch(p, nRange):
    branch = []
    pList = list(p)
    for i in nRange[::-1]:
        branch.append((i,pList[-i]))
    return branch

def dumpPickle(file, filename):
    if os.path.exists(filename):
        print('File already exists:', filename)
        return
    outfile = open(filename,'wb')
    pickle.dump(file, outfile)
    outfile.close()
    print('Pickle saved as:', filename)

def loadPickle(filename):
    print('Loading pickle:', filename)
    infile = open(filename,'rb')
    file = pickle.load(infile)
    infile.close()
    return file

def loadCombinations(N):
    filename = f'allCombinationsList_{N}.pickle'
    if os.path.exists(filename):
        allCombinations = loadPickle(filename)
    else:
        allCombinations = generateCombinations(N)
        dumpPickle(allCombinations, filename)
    return allCombinations


# =============================================================================
# Simulation
# =============================================================================


class Simulation():
    def __init__(self, simId:int, allCombinations:list, strategyNum:int=0):
        self.startTime = time.perf_counter()
        self.strategyNum = strategyNum
        self.statLog = []
        self.roundCounter = 0
        self.solutionFound = False
        self.elapsedTime = 0
        self.possCombis = allCombinations
        self.possCombisCount = len(allCombinations)
        self.uniquePairs = []
        self.uniquePairsCount = 0
        self.updatePairProbabilities()
        self.branches = 0
        print(f'SimId: {simId} | Combinations: {self.possCombisCount:,} | Unique pairs: {self.uniquePairsCount} | Strategy: {strategyNum}')


    def updatePairProbabilities(self):
        self.uniquePairs = list(set([y for x in self.possCombis for y in x]))
        self.pairCounts = dict(zip(self.uniquePairs,[0]*len(self.uniquePairs)))
        for combi in self.possCombis:
            for pair in combi:
                self.pairCounts[pair] += 1
        self.pairProbabilities = dict((k, v/self.possCombisCount) for k,v  in self.pairCounts.items())
        self.uniquePairsCount = len(self.pairCounts)


    def pairPossibleGain(self, pair, probTrue=1):
        baseValue = len(self.possCombis)
        valueTrue = len([x for x in self.possCombis if pair in x])
        valueFalse = len([x for x in self.possCombis if pair not in x])
        elimTrue = (baseValue - valueTrue) * probTrue
        elimFalse = (baseValue - valueFalse) * (1-probTrue)
        totalElimGain = elimTrue + elimFalse
        return totalElimGain



    def pickPair(self):
        if self.strategyNum==1:
            # Pick highest probability pair - randomizes among max prob pairs
            # otherwise keep going in one route
            maxProbPair = max(self.pairProbabilities, key=(lambda x: self.pairProbabilities[x]<1))
            maxProb = self.pairProbabilities[maxProbPair]
            pairsAtMax = [k for k,v in self.pairProbabilities.items() if v==maxProb]
            self.branches = len(pairsAtMax)
            pair = random.sample(pairsAtMax, 1)[0]
        elif self.strategyNum==2:
            # Test each pair to evaluate highest potential gain
            self.pairElimGain = {}
            for pair in self.uniquePairs:
                self.pairElimGain[pair] = self.pairPossibleGain(pair)
            pairAtMax = max(self.pairElimGain, key=self.pairElimGain.get)
            maxScore = self.pairElimGain[pairAtMax]
            pairsAtMax = [k for k,v in self.pairElimGain.items() if v==maxScore]
            self.branches = len(pairsAtMax)
            pair = random.sample(pairsAtMax, 1)[0]
        elif self.strategyNum==3:
            # Test each pair to evaluate highest potential gain using pair prob
            self.pairElimGain = {}
            for pair in self.uniquePairs:
                self.pairElimGain[pair] = self.pairPossibleGain(pair, self.pairProbabilities[pair])
            pairAtMax = max(self.pairElimGain, key=self.pairElimGain.get)
            maxScore = self.pairElimGain[pairAtMax]
            pairsAtMax = [k for k,v in self.pairElimGain.items() if v==maxScore]
            self.branches = len(pairsAtMax)
            pair = random.sample(pairsAtMax, 1)[0]
        elif self.strategyNum==4:
            maxEntropyPair = max(self.pairProbabilities, key=(lambda x: abs(self.pairProbabilities[x])))
            maxEntropy = abs(self.pairProbabilities[maxEntropyPair])
            pairsAtMax = [k for k,v in self.pairProbabilities.items() if abs(v)==maxEntropy]
            self.branches = len(pairsAtMax)
            pair = random.sample(pairsAtMax, 1)[0]
        else:
            # Random pick a pair that has non zero probability and is not already confirmed
            pair = random.sample([k for k,v in self.pairProbabilities.items() if ((v>0) & (v<1))], 1)[0]
        return pair


    def pickCombination(self):
        if self.strategyNum>0:
            maxScore = 0
            combisAtMax = []
            for combi in self.possCombis:
                score = sum([self.pairProbabilities[pair] for pair in combi])
                # When new maxScore found, reset combisAtMax list
                # If same as maxScore, keep combi for random sampling
                if score>maxScore:
                    maxScore = score
                    combisAtMax = [combi]
                elif score==maxScore:
                    combisAtMax.append(combi)
            combination = random.sample(combisAtMax, 1)[0]
            self.branches = len(combisAtMax)
#             combiAtMax = max(self.combiScoreMap, key=self.combiScoreMap.get)
#             maxScore = self.combiScoreMap[combiAtMax]
#             combisAtMax = [k for k,v in self.combiScoreMap.items() if v==maxScore]

        else:
            combination = random.sample(self.possCombis,1)[0]
        return combination


    def verifyPair(self, pair:tuple):
        return pair[0]==pair[1]

    def verifyCombination(self, combination:list):
        return len([1 for c in combination if c[0]==c[1]])


    def updateCombinations_pairInfo(self, pair:tuple, isMatch:bool):
        if isMatch:
            self.possCombis = [x for x in self.possCombis if pair in x]
        else:
            self.possCombis = [x for x in self.possCombis if pair not in x]
        self.possCombisCount = len(self.possCombis)
        self.updatePairProbabilities()


    def updateCombinations_combiInfo(self, tryCombi:list, correctPairs:int):
        passedCombis = []
        for assertTrueCombi in self.possCombis:
            if correctPairs==len([1 for c in assertTrueCombi if c in tryCombi]):
                passedCombis.append(assertTrueCombi)
        self.possCombis = passedCombis
        self.possCombisCount = len(self.possCombis)
        self.updatePairProbabilities()


    def runRound(self):
        self.roundCounter += 1
        pickedPair = self.pickPair()
        pairResult = self.verifyPair(pickedPair)
        prevPairsCount = self.uniquePairsCount
        prevCombisCount = self.possCombisCount
        pickedPairProb = self.pairProbabilities[pickedPair]
        self.updateCombinations_pairInfo(pickedPair, pairResult)
        self.statLog.append({'round': self.roundCounter,
                             'event': 'Pair',
                             'branches': self.branches,
                             'pickedPair': pickedPair,
                             'pickedPairProb': pickedPairProb,
                             'resultTB': pairResult,
                             'prevPairsCount': prevPairsCount,
                             'newPairsCount': self.uniquePairsCount,
                             'prevCombisCount': prevCombisCount,
                             'newCombisCount': self.possCombisCount,
                             })
        pickedCombi = self.pickCombination()
        numCorrect = self.verifyCombination(pickedCombi)
        prevPairsCount = self.uniquePairsCount
        prevCombisCount = self.possCombisCount
        pickedCombiScore = sum([self.pairProbabilities[pair] for pair in pickedCombi])
        self.updateCombinations_combiInfo(pickedCombi, numCorrect)
        self.statLog.append({'round': self.roundCounter,
                             'event': 'Combination',
                             'branches': self.branches,
                             'pickedCombi': pickedCombi,
                             'pickedCombiScore': pickedCombiScore,
                             'resultCeremony': numCorrect,
                             'prevPairsCount': prevPairsCount,
                             'newPairsCount': self.uniquePairsCount,
                             'prevCombisCount': prevCombisCount,
                             'newCombisCount': self.possCombisCount,
                             })

        if self.possCombisCount==1:
            self.solutionFound = True
            self.elapsedTime = time.perf_counter() - self.startTime




def runSimulation(simId, allCombinations, strategyNum):
    sim = Simulation(simId, allCombinations, strategyNum)
    ts = int(time.time())
    earlyTermination = False
    while not sim.solutionFound:
        sim.runRound()
        if sim.roundCounter>20:
            earlyTermination = True
            print(simId, 'Early termination')
            break
    df = pd.DataFrame(sim.statLog)
    df['combiChgPct'] = df['newCombisCount']/df['prevCombisCount']-1
    df['simId'] = str(strategyNum)+'_'+str(ts)+'_'+str(simId)
    df['elapsedTime'] = sim.elapsedTime
    df['earlyTermination'] = earlyTermination
    print(f'Sim: {simId} | Rounds: {sim.roundCounter} | Time: {sim.elapsedTime}')
    df.to_csv(f'./output/{strategyNum}_{ts}_{simId}.csv', index=False)
    return df


# =============================================================================
#
# =============================================================================

argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument('strategyNum', nargs=1, type=int)
argParser.add_argument('-m','--simulationRuns', nargs=1, type=int, default=100)
argParser.add_argument('-p','--parallel', action='store_true')

args = argParser.parse_args()

strategyNum = args.strategyNum[0]
m = args.simulationRuns[0]
run_parallel = args.parallel

print(f'Running with strategy {strategyNum} for {m} runs | Parallel: {run_parallel}')


N = 10
allCombinations = loadCombinations(N)
print(len(allCombinations))


if run_parallel:
    res = Parallel(n_jobs=6, verbose=5)(delayed(runSimulation)(simId, allCombinations, strategyNum) for simId in range(m))
else:
    for simId in range(m):
#         for strategyNum in range(4):
        runSimulation(simId, allCombinations, strategyNum)




