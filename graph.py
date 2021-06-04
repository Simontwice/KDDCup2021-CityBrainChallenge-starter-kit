import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


base_path = os.path.dirname(os.path.realpath(__file__))
files_path = os.path.join(base_path, 'bests_so_far/graphs')

print(files_path)
files_path = os.path.join(files_path,'*')
files = sorted(
    glob.iglob(files_path), key=os.path.getctime, reverse=True)


d_indices = np.genfromtxt(files[1])
differences = np.genfromtxt(files[0])

fig,ax = plt.subplots(nrows=2)

how_many = 7
for i in range(how_many):
    try:
        ax[0].plot(np.genfromtxt(files[2*i+1])[0:5], '.-',label = str(i),)
    except:
        pass
    ax[1].plot(np.genfromtxt(files[2*i]), label=str(i))
ax[0].legend()
ax[0].set(xlabel = 'iteration step (every 200)', ylabel = 'delay index',title = 'delay index evolution over time')
ax[1].legend()
ax[1].set(xlabel = 'iteration step (every 200)', ylabel = 'delay index increase',title = 'delay index increase over time')
plt.tight_layout()
matplotlib.pyplot.show()
