import numpy as np
import tensorflow as tf
import random

nodeCnt = 0
arcCnt = 0
arcList = []
pathList = []
pathRatio = []
floyd = []

def loadGraph(fileIn, _scaleCapac):
    with open(fileIn, "r") as f:
        data = f.readline().split()
        global nodeCnt, arcList, arcCnt, pathList, floyd
        nodeCnt = int(data[0])
        edgeCnt = int(data[1])
        arcCnt = edgeCnt * 2
        arcList = [[0 for i in range(3)] for j in range(edgeCnt)]
        floyd = [[100 for i in range(nodeCnt) ] for j in range(nodeCnt)]
        pathList = [[ [0] for i in range(nodeCnt)] for j in range(nodeCnt)]
        for i in range(edgeCnt):
            data = f.readline().split()
            arcList[i][0] = int(data[0]) - 1
            arcList[i][1] = int(data[1]) - 1
            arcList[i][2] = int(data[3]) * _scaleCapac
            
            x = arcList[i][0]
            y = arcList[i][1]
            floyd[x][y] = 1
            floyd[y][x] = 1
            pathList[x][y][0] = 1
            pathList[x][y].append( [i] )
            pathList[y][x][0] = 1
            pathList[y][x].append( [i+edgeCnt] )
        for i in range(edgeCnt):
            arc = [ arcList[i][1], arcList[i][0], arcList[i][2] ]
            arcList.append(arc)
    return

#It's a specialized floyd, I love this idea!
def runFloyd():
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (j == i):
                continue
            for l in range(nodeCnt):
                if ((l == i) or (l == j)):
                    continue
                newDis = floyd[j][i] + floyd[i][l]
                if (newDis < floyd[j][l]):
                    floyd[j][l] = newDis
                    pathList[j][l] = [0]
                if (newDis == floyd[j][l]):
                    for p1 in range(pathList[j][i][0]):
                        for p2 in range(pathList[i][l][0]):
                            newPath = pathList[j][i][p1+1] + pathList[i][l][p2+1]
                            pathList[j][l][0] += 1
                            pathList[j][l].append(newPath)
                            
def setOSPFMod(mod):
    global pathRatio
    pathRatio = [ [[] for i in range(nodeCnt)] for j in range(nodeCnt)]
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            pathRatio[i][j].append( pathList[i][j][0] )
            if (mod == 1):
                for l in range(pathRatio[i][j][0]):
                    pathRatio[i][j].append(0)
                p =  random.randint(1, pathRatio[i][j][0])
                pathRatio[i][j][p] = 1
            else:
                for l in range(pathRatio[i][j][0]):
                    pathRatio[i][j].append( 1.0 / pathRatio[i][j][0])

def solveOSPF(traMat):
    loadList = [0 for i in range(arcCnt)]
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            for l in range(pathList[i][j][0]):
                p = l + 1
                for arc in pathList[i][j][p]:
                    loadList[arc] += traMat[i][j] * pathRatio[i][j][p]
    Maxx = 0
    for i in range(arcCnt):
        if (Maxx < (loadList[i] / arcList[i][2])):
            Maxx = (loadList[i] / arcList[i][2])
    return Maxx 
                   
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
                    traMat[x][y] = float(tra[i])
                    y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                target = solveOSPF(traMat)
                setOSPFMod(1)
                print(target, file = fout)
    return
    
if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('graphFile', "", "")
    flags.DEFINE_string('tmFile'   , "", "")
    flags.DEFINE_string('perfFile' , "", "")
    flags.DEFINE_float('scaleCapac' , 1, "")

    FLAGS = flags.FLAGS
    _graphFile = getattr(FLAGS, "graphFile", None)
    _tmFile    = getattr(FLAGS, "tmFile"   , None)
    _perfFile  = getattr(FLAGS, "perfFile" , None)
    _scaleCapac = getattr(FLAGS, "scaleCapac")
    print("_scaleCapac is : ", _scaleCapac)

    if (_graphFile != ""):
        loadGraph(_graphFile, _scaleCapac)
    runFloyd()
    setOSPFMod(1)
    if (_tmFile != ""):
        solvePerTM(_tmFile)
    