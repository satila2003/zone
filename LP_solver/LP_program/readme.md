# 线性规划最终使用版

第一版本的程序把各个函数杂糅到了一起，用起来很不方便，因此我把它进行了拆分，在这个文件夹中的对应的3个sh能通过配置对应的参数来简单快速地计算utl!

## 一些通用的定义

由于3个程序复用了一些模块，因此这里需要额外阐述一些格式细节：

graphFile描述拓扑时，节点需要从1开始编号，同时每条边的描述包含4个整数，分别为两个端点，没啥用的占位数字和这条边的流量

pathFile描述备选路径，节点从0开始编号，需要以succeed结尾

tmFile描述流量矩阵，节点从0开始编号，需要以succeed开始（所以建议和pathFile定义为同一个文件，避免问题）

perfFile则是输出文件，每行表示对应TM上的utl

## `runAltPath.sh`

runAltPath通过调用altPathSolver计算给定流量矩阵的情况下备选路径策略所能达到的最优解。

graphFile描述拓扑，pathFile描述备选路径，tmFile描述流量矩阵，perfFile描述输出，和通用定义完全一致

## `runSeer.sh`

runSeer通过调用seerSolver计算给定流量矩阵的情况下全局路径可选所能达到的最优解。（即最终的，不可超越的极限）

graphFile描述拓扑，tmFile描述流量矩阵，perfFile描述输出，由于没有备选路径的概念，就不需要pathFile咯！
