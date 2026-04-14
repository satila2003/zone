#!/bin/bash -x

numTM=100
topoName=NEW_6_6
method=SHR
dim=trueTM
tvt=test
scaleCapac=200.0

# python3 ospfSolver.py \
# --graphFile=../originInput/Abi/Abi.txt \
# --tmFile=../originInput/Abi/Abi_trueTM_OR_test48000.txt \
# --perfFile=../LP_output/Util_Abi_ospf_48000.txt \
# --scaleCapac=50

python3 ospfSolver.py \
--graphFile=../originInput/${topoName}/${topoName}.txt \
--tmFile=../originInput/${topoName}/${topoName}_${dim}_${method}_${tvt}${numTM}.txt \
--perfFile=../LP_output/${topoName}/Util_${topoName}_${method}_${numTM}.txt \
--scaleCapac=${scaleCapac}

# python3 ospfSolver.py \
# --graphFile=../originInput/NTELOS.txt \
# --tmFile=../originInput/NTELOS_OR_tm2000.txt \
# --perfFile=../LP_output/Util_NTELOS_ospf_2000.txt \
# --scaleCapac=5