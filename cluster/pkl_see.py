import pickle
import pandas as pd
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms/Labels_starlink550_cluster.pkl'
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms\starlink550_cluster.pkl'
# path = 'F:\Py_Project/always\Build_datasets\Labels_Iridium_Max_Throughput.pkl'

# path = 'F:\Py_Project/always\Build_datasets\Labels_Iridium_Optimized.pkl'
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms/weights_slice_0_domain_None.pkl'
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms/slice_0_domain_0.pkl'
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms\starlink550_cluster.pkl'
# path = 'F:\Py_Project/always\cluster\zone\outputs/tms/starlink550.pkl'
# path = 'F:\Py_Project/always\Build_datasets\StarLink_DataSetForAgent100_5000_A.pkl'
path = 'F:\Py_Project/always\cluster\zone\outputs/tms/starlink550_intra.pkl'

f = open(path, 'rb')
data = pickle.load(f)
# print(data)
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
