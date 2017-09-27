
import scipy.io
import numpy as np

data = scipy.io.loadmat("hm2data.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        np.savetxt(("filesforyou/"+i+".csv"),data[i],delimiter=',')