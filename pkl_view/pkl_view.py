import pickle

# path = '../traffic_matrices/abilene/t1.pkl' #132=12*11个流量需求
# path = '../pairs/abilene/t1.pkl'
# path = '../pairs/geant/t1.pkl' #22个节点对
path = 'F:\Py_Project/always\Build_datasets\StarLink_Labels_Iridium.pkl' #462=22*21个流量需求
# path = 'F:\Py_Project/always\Build_datasets\StarLink_DataSetForAgent_Iridium.pkl'
# path = 'F:\Py_Project/always\Build_datasets\iridium_ksp_paths/iridium_200steps_paths.pkl'
f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))
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
