import os
import sys
sys.path.append('/home/guifei/software_packages/gurobi811/linux64/lib/python3.7_utf32/gurobipy')

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import tensorflow as tf

import math

nodeCnt = 0
edgeCnt = 0
arcCnt = 0
edgeList = []
pathList = []
traMat = []

scaleCapac = 5 # WIDE
#scaleCapac = 50 #Abi

def stToArc(src, tag):
    global arcCnt, edgeList
    for i in range(arcCnt):
        if ((edgeList[i][0] == src) and (edgeList[i][1] == tag)):
            return i
    return

def ijToRank(i, j):
    global nodeCnt
    rank = i * (nodeCnt - 1) + j
    if (j > i):
        rank -= 1
    return 3 * rank

def loadGraph(fileIn):
    with open(fileIn, "r") as f:
        data = f.readline().split()
        global nodeCnt, edgeCnt, edgeList, arcCnt
        nodeCnt = int(data[0])
        edgeCnt = int(data[1])
        arcCnt = edgeCnt * 2
        edgeList = [[0 for i in range(3)] for j in range(edgeCnt)]
        for i in range(edgeCnt):
            data = f.readline().split()
            edgeList[i][0] = int(data[0]) - 1
            edgeList[i][1] = int(data[1]) - 1
            edgeList[i][2] = int(data[3]) * scaleCapac

        for i in range(edgeCnt):
            edge = [ edgeList[i][1], edgeList[i][0], edgeList[i][2] ]
            edgeList.append(edge)
    return
    
def loadPath(fileIn):
    with open(fileIn, "r") as f:
        global edgeCnt, nodeCnt, pathList
        pathList = [[[0] for i in range(nodeCnt)] for j in range(nodeCnt)]
        for line in f.readlines():
            if (line[:4] == 'succ'):
                break
            else:
                if (line.find(',') == -1):
                    continue
                numList = line.split(',')
                del(numList[0])
                del(numList[-1])
                for i in range(len(numList)):
                    numList[i] = int(numList[i])
                src = numList[0]
                tag = numList[-1]
                pathList[src][tag][0] += 1
                path = []
                for i in range(len(numList) -1):
                    path.append(stToArc(numList[i], numList[i+1]))
                pathList[src][tag].append(path)
    return
    
def solveAltPath(traMat):
    global nodeCnt, edgeCnt, pathList
    
    A_ub = []
    Vcnt = nodeCnt * (nodeCnt - 1) * 3
    
    model = gp.Model('AltPath')
    model.setParam('OutputFlag', 0)
    
    vList = []
    
    for i in range(Vcnt):
        vList.append(model.addVar(0.0, 1.0, name=str(i)))
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r")
    model.setObjective(r, GRB.MINIMIZE)
    
    #CONS1, the PERF R
    for i in range(arcCnt):
        A_ub.append(gp.LinExpr())
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            for l in range(pathList[i][j][0]):
                path = pathList[i][j][l+1]
                for edge in path:
                    A_ub[edge] += ( (traMat[i][j]/edgeList[edge][2]) * vList[ijToRank(i,j)+l] )
    for i in range(arcCnt):
        model.addConstr(A_ub[i] <= r)
    
    #CONS2, the RATIO
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            Aside = gp.LinExpr()
            for l in range(pathList[i][j][0]):
                Aside += vList[ ijToRank(i, j)+l ] 
            model.addConstr(Aside == 1.0)
            
            for l in range(pathList[i][j][0], 3):
                Aside = gp.LinExpr( vList[ ijToRank(i,j)+l ] )
                model.addConstr(Aside == 0.0)

    model.optimize()
    target = model.objVal
    solution = []
    for i in range(Vcnt):
        solution.append( model.getVarByName(str(i)).X)
    return target, solution
    
def solvePerTM(fileIn):

    lineCnt = 0
    fout = open(_perfFile, "w")
        
    with open(fileIn, "r") as f:
        global nodeCnt, traMat
        flag = True
        for line in f.readlines():
            if (line[:4] == 'succ'):
                flag = False
            else:
                if (flag):
                    continue
                if (line.find(',') == -1):
                    continue
                lineCnt += 1
                traMat = [[0 for i in range(nodeCnt)] for j in range(nodeCnt)]
                tra = line.split(',')
                x = 0
                y = 0
                for i in range(nodeCnt * nodeCnt - nodeCnt):
                    if (x == y):
                        y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                    traMat[x][y] = math.ceil(float(tra[i]))
                    y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                target, solution = solveAltPath(traMat)
                print(target, file = fout)
                print(target)
    return
    
if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('graphFile', "", "")
    flags.DEFINE_string('pathFile' , "", "")
    flags.DEFINE_string('tmFile'   , "", "")
    flags.DEFINE_string('perfFile' , "", "")

    FLAGS = flags.FLAGS
    _graphFile = getattr(FLAGS, "graphFile", None)
    _pathFile  = getattr(FLAGS, "pathFile" , None)
    _tmFile    = getattr(FLAGS, "tmFile"   , None)
    _perfFile  = getattr(FLAGS, "perfFile" , None)

    print("for ", _pathFile)
    
    if (_graphFile != ""):
        loadGraph(_graphFile)
    if (_pathFile != ""):
        loadPath(_pathFile)
    if (_tmFile != ""):
        solvePerTM(_tmFile)
