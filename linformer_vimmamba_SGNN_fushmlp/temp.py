import torch
import numpy as np 
import os
import pandas  as  pd
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

path = '/home/lxy/programs/lxy_new/dataset/des.pkl'

with open (path,'rb') as f :
    data = pickle.load(f)

destination = np.array(list(data.values()))
dest= destination[:,-1,]
mykmeans = KMeans(n_clusters=80)
mykmeans.fit(dest)
centers = mykmeans.cluster_centers_
plt.figure(dpi=600)
for x,y in dest[:2000]:
    plt.scatter(x,y,s=0.2,color = 'blue')
for x1,y1 in centers:
    plt.scatter(x1,y1,s=0.5,color = 'red')
plt.scatter(0,0,s=1,color='green')
# for a in destination[:10]:
    
#     plt.plot(a[20:,0],a[20:,1],linewidth=0.5,color='blue')
plt.show()


# with open(os.path.join('/home/lxy/programs/lxy_new/dataset','kmeans_center.pkl_80'),'wb') as fp:
#     pickle.dump(centers,fp)


# import torch
# import numpy as np 
# import os
# import pandas  as  pd
# from tqdm import tqdm
# import pickle
# from sklearn.cluster import KMeans

# path = '/home/lxy/argo1_data/argoverse1/train/data'
# files = os.listdir(path)
# # os.makedirs('/home/lxy/programs/lxy_new/dataset/des.pkl')
# des = {}
# # plt.figure(dpi=600)
# for file in tqdm(files):
#     file_path = os.path.join(path,file)
#     with open(file_path,'rb') as f:
#         df = pd.read_csv(f)
#     objs = df.groupby(['TRACK_ID','OBJECT_TYPE']).groups
#     keys = list(objs.keys())
#     obj_type = [x[1] for x in keys]
#     agt_idx = obj_type.index('AGENT')
#     idcs = objs[keys[agt_idx]]
#     trajs = np.concatenate((
#             df.X.to_numpy().reshape(-1,1),
#             df.Y.to_numpy().reshape(-1,1)),1)
    
#     agt_traj = trajs[idcs]
#     orig = agt_traj[19]
#     pre = orig-agt_traj[18]
#     theta = np.arctan2(pre[1],pre[0])
#     rot = np.asarray([
#                          [np.cos(theta),-np.sin(theta)],
#                          [np.sin(theta),np.cos(theta)]],np.float32)
#     agent_traj =np.dot(agt_traj - orig, rot)
#     # plt.plot(agent_traj[20:,0],agent_traj[20:,1],linewidth=0.3,color='blue')
# # plt.show()
#     des[file[:-4]] = agent_traj
# # destination = des.values()

# with open(os.path.join('/home/lxy/programs/lxy_new/dataset','des.pkl'),'wb') as fp:
#     pickle.dump(des,fp)
