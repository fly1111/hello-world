
# coding: utf-8

# In[2]:




import pickle
import numpy as np
import os

CIFAR_DIR = "./cifar-10-batches-py"
print (os.listdir(CIFAR_DIR))


# In[4]:


with open(os.path.join(CIFAR_DIR,"data_batch_1"), 'rb') as f:
    data = pickle.load(f,encoding='bytes')
    print (type(data))
    print (data.keys())
    print (type(data[b'data']))
    print (type(data[b'labels']))
    print (type(data[b'batch_label']))
    print (type(data[b'filenames']))
    print (data[b'data'].shape)
    print (data[b'data'][0:2])
    print (data[b'filenames'][0:3])
    


# In[13]:


image_arr = data[b'data'][101]
image_arr = image_arr.reshape((3,32,32)) #32 32 3
image_arr = image_arr.transpose((1,2,0))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.imshow(image_arr)

