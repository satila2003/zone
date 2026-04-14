import pickle
import pandas as pd
# path = '../traffic_matrices/abilene/t1.pkl' #132=12*11个流量需求
# path = 'F:\Py_Project/always\Build_datasets\Iridium_DataSetForAgent_75_60480.pkl'
# path = 'F:\Py_Project/always\Build_datasets\StarLink_DataSetForAgent_Iridium.pkl'
# path = 'F:\Py_Project/always\Build_datasets\Gurobi_size-66_mode-NA_intensity-75_volume-5000_solutions.pkl'
# path = 'F:\Py_Project/always\Build_datasets\iridium_ksp_paths/iridium_200steps_paths.pkl'
# path = 'F:\Py_Project/always\Build_datasets\Labels_Iridium_Optimized.pkl'
# path = 'F:\Py_Project/always\Build_datasets\Iridium_DataSetForAgent_75_60480.pkl'
# path = 'F:\Py_Project/always\Build_datasets\Labels_Iridium_Max_Throughput.pkl'

path = 'F:/Py_Project\cluster\zone\outputs/tms\step_0492_2026-03-25_02-11-26.pkl' #132=12*11个流量需求
f = open(path, 'rb')
data = pickle.load(f)
# 下面这一行用来查看list
# df = pd.DataFrame(data[0])
# print(data)
# print(len(data))
# print(data.shape())


#
# import torch
# import numpy as np
# # path = '../model/HARP_abilene_pred_1_8sp.pkl'
# path = '../topologies/paths_dict/abilene_8_paths_dict_cluster_0.pkl'
# f = open(path, 'rb')
# data = torch.load(f, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 可使用cpu或gpu
# data = (data.to_dense()).cpu()  # 转换为密集张量
# data=np.array(data)
# #使用numpy查看
# # print(np.array(data))
# print(data)
# print(data.shape)
