#!/bin/bash
# 绝对路径示例
PYTHON="D:/Anaconda/envs/redTE/python.exe"

### Abi
$PYTHON seerSolver.py \
--graphFile=../originInput/Abi/Abi.txt \
--tmFile=../originInput/Abi/Abi_OR_tm500.txt \
--perfFile=../LP_output/Abi/Util_Abi_seer_500.txt \
--spatioFile=../LP_output/Abi/Spatio_Abi_seer_500.txt \
--scaleCapac=50  #the scaling factor of link bandwidth

### GEA
$PYTHON seerSolver.py \
--graphFile=../originInput/GEA/GEA.txt \
--tmFile=../originInput/GEA/GEA_OR_tm500.txt \
--perfFile=../LP_output/GEA/Util_GEA_seer_500.txt \
--spatioFile=../LP_output/GEA/Spatio_GEA_seer_500.txt \
--scaleCapac=625

