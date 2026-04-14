# 线性规划
# 想要计算LP对应的utl， seerSolver计算任意路径可选情况下的最优解

# 注意事项
这里的流量矩阵格式应该是嵌套的列表，而且元素个数是节点数 * 节点数（对角的流量虽然无关紧要，但必须要有），譬如 [ [100 for i in range(nodeCnt)] for j in range(nodeCnt)]

# 如何跑起来
第一步，环境准备：需要安装gurobi（我们装的是8.1.1）

然后在打开LP_solver/LP_program/seerSolver.py, 修改gurobipy的路径（在gurobi的安装路径下）

sys.path.append('/home/guifei/software_packages/gurobi811/linux64/lib/python3.7_utf32/gurobipy')

## 进入路径：
cd ./LP_solver/LP_program/

## 运行：
sh runSeer.sh

## 注意： 
runSeer.sh的变量描述：
graphFile： 拓扑输入文件
tmFile： TM流量矩阵输入文件
perfFile： 计算结果（最大链路利用率）
scaleCapac： 链路利用率缩放比，和训练代码的Env1110.py保持一致；
