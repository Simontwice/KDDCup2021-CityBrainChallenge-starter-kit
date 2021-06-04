import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("/home/szymon/Downloads/KDDCup2021-CityBrainChallenge-starter-kit/traffic_shape")
filelist = os.listdir(".")
num_of_files = len(filelist)
counts = np.zeros((12,1))
magnitudes = []
num_lanes_used = []
def meaningful_lanes(boi2):
    boi=np.ravel(boi2)
    boi_sum = np.sum(boi)
    if boi_sum == 0:
        boi_sum=1
    boi = boi/boi_sum
    boi = -np.sort(-boi)
    boi=np.cumsum(boi)
    boi[boi>0.99]=0
    boi[0]=1
    return np.sum(boi>0)

for file in filelist:
    a = np.loadtxt(file)
    a = np.sum(a,axis=-1).reshape(12,-1)
    counts = np.concatenate((counts,a),axis=-1)
    a[a<0]=0
    num_lanes_used.append(meaningful_lanes(a))
    magnitudes.append(np.sum(a))
print(counts.shape)
print(num_lanes_used)
#plt.hist(magnitudes,bins=30)
#plt.show()
#plt.hist(num_lanes_used,bins= [0,1,2,3,4,5,6,7,8,9,10,11,12])
#plt.show()
plt.hist2d(magnitudes,num_lanes_used,bins= (np.linspace(0,2000,10),[0,1,2,3,4,5,6,7,8,9,10,11,12]))

plt.colorbar()
plt.show()
