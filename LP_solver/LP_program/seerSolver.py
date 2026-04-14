import os
import sys

import gurobipy as gp
from gurobipy import GRB

import numpy as np
import math
import tensorflow as tf
#import flag

nodeCnt = 0 # 节点数
edgeCnt = 0 # 无向边数
arcCnt = 0 # 有向弧数
edgeList = [] # 存放边/弧的列表

sePath = [] # 用来保存解出来的路径
seMat = [] # 用来保存解出来的流量分配矩阵
flowRemain = [] # 用来保存剩余流量矩阵
MINN = 0.000001



def RDFS(now, path):
    start = path[0] # 起点
    for i in range(nodeCnt):
        if (flowRemain[now][i] < MINN): # 无剩余流量，跳过
            continue
        if (i != start): # 不是起点
            if (i in path): # 出现在当前路径（环），跳过
                continue
            path.append(i)
            RDFS(i, path)
            del(path[-1])
        else: # 回到起点，找到环
            tmp = flowRemain[now][i] 
            for j in range(len(path)-1): # 找到路径中最小的剩余流量
                tmp = min(tmp, flowRemain[ path[j] ][ path[j+1] ])
            flowRemain[now][i] -= tmp # 减去环上最小的剩余流量
            for j in range(len(path)-1): 
                flowRemain[ path[j] ][ path[j+1] ] -= tmp 
            
def removeLoop(src, tag):
    for i in range(nodeCnt):
        for j in range(nodeCnt): # 找到双向边上的最小剩余流量，减去
            tmp = min(flowRemain[i][j], flowRemain[j][i])
            flowRemain[i][j] -= tmp
            flowRemain[j][i] -= tmp
    for i in range(nodeCnt): # 从源点出发，深度优先遍历，移除环路
        RDFS(i, [i])
    return
    
def DFS(tar, path, flow):
    global MINN, sePath, flowRemain
    
    now = path[-1] # 当前在路径的最后一个节点
    if (flow < MINN): # 流量过小，停止
        return
    if (now == tar): # 到达目标节点，保存路径和流量
        sePath.append([])
        for node in path:
            sePath[-1].append(node)
        sePath[-1].append(flow)
        return
        
    for i in range(nodeCnt):
        if (i in path): # 避免环路
            continue
        newflow = min(flowRemain[now][i], flow) # 当前边的剩余流量和当前流量的最小值
        if (newflow < MINN): # 无剩余流量，跳过
            continue
        flow -= newflow # 把这部分流量分配出去
        flowRemain[now][i] -= newflow # 减去这部分流量
        path.append(i) # 把当前节点加入路径
        DFS(tar, path, newflow) # 递归继续深度优先搜索
        del(path[-1]) # 回溯，移除当前节点
    return
    
def decodePath():
    global flowRemain, sePath
    num = 0
    pathCnt = 0
    
    sePath = []
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j): # 跳过相同节点
                continue
            flowRemain = [ [0 for x in range(nodeCnt)] for y in range(nodeCnt)]
            for l in range(arcCnt):
                flowRemain[edgeList[l][0]][edgeList[l][1]] = float(seMat[num*arcCnt + l]) # 按节点进行流量赋值
            num = num + 1
            removeLoop(i, j) # 移除环路
            DFS(j, [i], 1.0) # 深度优先记录路径并分配流量
    
    return sePath

def loadGraph(fileIn, scaleCapac):
    with open(fileIn, "r") as f:
        data = f.readline().split()
        global nodeCnt, edgeCnt, edgeList, arcCnt
        nodeCnt = int(data[0]) 
        edgeCnt = int(data[1])
        arcCnt = edgeCnt * 2
        edgeList = [[0 for i in range(3)] for j in range(edgeCnt)]
        for i in range(edgeCnt): # 正向边
            data = f.readline().split()
            edgeList[i][0] = int(data[0]) - 1 # 起点
            edgeList[i][1] = int(data[1]) - 1 # 终点
            edgeList[i][2] = int(data[3]) * scaleCapac # 容量，乘以缩放系数
        for i in range(edgeCnt): # 反向边
            edge = [ edgeList[i][1], edgeList[i][0], edgeList[i][2] ]
            edgeList.append(edge)
    return

def ijaToRank(i, j, arc): # 把“源节点 i、目标节点 j、弧编号 arc”映射成一个连续的一维索引 rank
    global nodeCnt, arcCnt # 节点数，有向弧数
    rank = i * (nodeCnt - 1) + j
    if (j > i):
        rank -= 1
    rank = rank * arcCnt + arc
    return rank

def solveSeer(traMat, decode):
    global nodeCnt, edgeCnt, edgeList
    global seMat
    
    A_ub = []
    Vcnt = nodeCnt * (nodeCnt-1) * arcCnt # 所有 (i,j,arc) 变量的总数
    vList = []
    
    model = gp.Model('Seer')
    # —— Add these right after model creation ——
    # Force single thread and record a log to investigate crash
    try:
        model.setParam('Threads', 1)                         # force single-thread
        # model.setParam('LogFile', 'gurobi_debug.log')        # create log file
        # optional: try different methods if single-thread doesn't help
        # model.setParam('Method', 1)   # primal simplex
        # model.setParam('Method', 2)   # dual simplex
        # model.setParam('Method', 4)   # barrier
    except Exception as e:
        print("WARNING: failed to set gurobi params:", e)

    # Before optimize, print model size info
    print("DEBUG: Before optimize, NumVars =", model.NumVars, "NumConstrs =", model.NumConstrs)

    model.setParam('OutputFlag', 0)
    
    seMat = []
    seMat = [0 for i in range(Vcnt)]
    for i in range(Vcnt):
        vList.append(model.addVar(0.0, 1.0, name=str(i))) # 添加变量，范围0到1
    r = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r") # 添加变量 r，表示最大利用率
    model.setObjective(r, GRB.MINIMIZE) # 目标函数：最小化 r
    
    #CONS1, the PERF R
    for i in range(arcCnt):
        A_ub.append(gp.LinExpr()) # 初始化线性表达式列表
    
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j): # 跳过相同节点
                continue
            for arc in range(arcCnt): # 如果把 traMat[i][j] 全放在这条弧会占用多少比例
                A_ub[arc] += ( (traMat[i][j] / edgeList[arc][2]) * vList[ijaToRank(i,j,arc)] )
    for i in range(arcCnt):
        model.addConstr(A_ub[i] <= r) # 添加约束，确保每条弧的利用率不超过 r
    
    #CONS2, the RATIO
    for i in range(nodeCnt):
        for j in range(nodeCnt):
            if (i == j):
                continue
            for l in range(nodeCnt):
                Aside = gp.LinExpr()
                for arc in range(arcCnt):
                    if (edgeList[arc][0] == l):    #Sub Out flow，减去流出
                        Aside -= vList[ ijaToRank(i,j,arc) ]
                    if (edgeList[arc][1] == l):    #Add In flow，加上流入
                        Aside += vList[ ijaToRank(i,j,arc) ]
                    #How many flow remained 
                if (l == i): # 源节点，流出为-1
                    model.addConstr(Aside == -1.0)
                if (l == j): # 汇节点，流入为1
                    model.addConstr(Aside == 1.0)
                if ((l != j) and (l != i)): # 中间节点，流入流出相等
                    model.addConstr(Aside == 0.0)
                     
    model.optimize()
    
    strategy = None
    if (decode): # 解码路径
        for i in range(Vcnt):
            num = model.getVarByName(str(i)).X
            seMat[i] = num 
            # print(seMat[i])
        strategy = decodePath()
    return (model.objVal),strategy # 返回目标值和sePath
    
def solvePerTM(fileIn):
    lineCnt = 0
    foutmlu = open(_perfFile, "w")
    foutspatio = open(_spatioFile, "w")
        
    with open(fileIn, "r") as f:
        global nodeCnt, traMat
        flag = True
        for line in f.readlines():
            if (line[:4] == 'succ'): # 从succeed标志下一行开始读TM
                flag = False
            else:
                if (flag):
                    continue
                if (line.find(',') == -1): # 如果没有逗号，不是TM行，跳过
                    continue
                lineCnt += 1 # 统计读取的流量需求行数
                traMat = [[0 for i in range(nodeCnt)] for j in range(nodeCnt)]
                tra = line.split(',')
                x = 0
                y = 0
                for i in range(nodeCnt * nodeCnt - nodeCnt):
                    if (x == y): # 跳过对角线
                        y += 1
                    if (y == nodeCnt): # 到达行尾，换行
                        x += 1
                        y = 0
                    traMat[x][y] = math.ceil(float(tra[i])) # 四舍五入取整
                    y += 1
                    if (y == nodeCnt):
                        x += 1
                        y = 0
                target, solution = solveSeer(traMat, True)
                print(target, file = foutmlu)
                print(solution,file = foutspatio)
                print(lineCnt)
                print(target)
    return


# scaleCapac = 5 # WIDE
# scaleCapac = 625 # GEA
# scaleCapac = 50 # Abi

if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string('graphFile', "", "")
    flags.DEFINE_string('tmFile'   , "", "")
    flags.DEFINE_string('perfFile' , "", "")
    flags.DEFINE_string('spatioFile' , "", "")
#    flags.DEFINE_integer('scaleCapac' , 1, "")

    flags.DEFINE_float('scaleCapac' , 1, "")

    FLAGS = flags.FLAGS
    _graphFile = getattr(FLAGS, "graphFile", None)
    _tmFile    = getattr(FLAGS, "tmFile"   , None)
    _perfFile  = getattr(FLAGS, "perfFile" , None)
    _spatioFile  = getattr(FLAGS, "spatioFile" , None)
    _scaleCapac = getattr(FLAGS, "scaleCapac")
    print("_scaleCapac is : ", _scaleCapac)

    if (_graphFile != ""):
        loadGraph(_graphFile, _scaleCapac)
    if (_tmFile != ""):
        solvePerTM(_tmFile)
