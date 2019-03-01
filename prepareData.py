import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd


# Define the data dimensions
# images are 128 x 118 with 1 channel of color (gray),_c and _v
input_dim_model = (2, 128, 118)
input_dim_row = 128
input_dim_col = 118
# used by readers to treat input data as a vector
input_dim = 128*118
train_base_path = os.path.join("Training_data", "train")
img_dtype = np.float64
#train_path = 'train'
num_train_samples = 988
num_predict_samples = 659


#0: Compact Fluorescent Lamp
#1: Hairdryer
#2: Microwave
#3: Air Conditioner
#4: Fridge
#5: Laptop
#6: Vacuum
#7: Incandescent Light Bulb
#8: Fan
#9: Washing Machine
#10: Heater
num_output_classes = 11


#read data
def readImages():
    data_c = np.zeros((num_train_samples, input_dim), dtype=img_dtype)
    data_v = np.zeros((num_train_samples, input_dim), dtype=img_dtype)
    for img_id in range(num_train_samples):
        #_c.png
        img_c_name = str(1000 + img_id) + "_c.png"
        img_c_file = os.path.join("Training_data", "train", img_c_name)
        imge_c = io.imread(img_c_file, as_grey=True)
        data_c[img_id, :] = imge_c.flatten()
        #_v.png
        img_v_name = str(1000 + img_id) + "_v.png"
        img_v_file = os.path.join("Training_data", "train", img_v_name)
        imge_v = io.imread(img_v_file, as_grey=True)
        data_v[img_id, :] = imge_v.flatten()
    print("1111111111111111")
    print(data_c[0, :])
    #print(train_data[0,:].reshape(2,128,118))
    print("1111111111111111")
    train_data = np.hstack((data_c, data_v))

    return train_data

#read test data
def readPredictImages():
    data_c = np.zeros((num_predict_samples, input_dim), dtype=img_dtype)
    data_v = np.zeros((num_predict_samples, input_dim), dtype=img_dtype)
    #_c.png
    for img_id in range(num_predict_samples):
        img_name = str(1988 + img_id) + "_c.png"
        img_file = os.path.join("Training_data", "test", img_name)
        imge = io.imread(img_file, as_grey=True)
        data_c[img_id, :] = imge.flatten()
    #_v.png
    for img_id in range(num_predict_samples):
        img_name = str(1988 + img_id) + "_v.png"
        img_file = os.path.join("Training_data", "test", img_name)
        imge = io.imread(img_file, as_grey=True)
        data_v[img_id, :] = imge.flatten()

    predict_data = np.hstack((data_c, data_v))

    return predict_data

def OneHotEncoder(labels, classNum=num_output_classes):
    """
    """
    #_cond = np.array([list(range(classNum)), ] * labels.shape[0])
    #cond = _cond == labels.reshape(-1, 1)
    oneHot = np.zeros((labels.shape[0], classNum))
    for row in range(labels.shape[0]):
        col = labels[row]
        oneHot[row, col] = 1
    return oneHot

#load data
def loadData():
    lables_file = pd.read_csv(os.path.join("Training_data", "train_labels.csv"))
    labels = np.array(lables_file["appliance"])

    trainImg = readImages()
    testImg = readPredictImages()
    trainLabel = OneHotEncoder(labels)

    return trainImg, trainLabel, testImg
'''
#data_c = np.zeros((num_train_samples, input_dim_row*input_dim_col),dtype = np.uint8)
#label_c = np.zeros((num_train_samples, 1),dtype = np.uint8)
#img1 = io.imread("1000_c.png", as_grey=True)
#print(type(img1))
#print(img1.shape)
#for img_id in range(num_train_samples):
    img_name = str(1000+img_id) + "_c.png"
    img_file = os.path.join("Training_data", "train", img_name)
    imge_c = io.imread(img_file, as_grey=True)
    data_c[img_id,:] = imge_c.flatten()

#print(os.path.join("Training_data", "train_labels.csv"))
lables = pd.read_csv(os.path.join("Training_data", "train_labels.csv"))
print(lables.shape)
print(lables["appliance"].head(5))
print(np.array(lables["appliance"]).ndim)

label_c = (np.array(lables["appliance"])).reshape(num_train_samples, 1)
train_c = np.hstack((data_c, label_c))

# Plot a random image
sample_number = 0
plt.imshow(train_c[sample_number,:-1].reshape(128,118), cmap="gray_r")
#print(train_c[sample_number,:-1])
plt.axis('off')
print("Image Label: ", train_c[sample_number,-1])
'''

'''
img_c_name = str(1000 + 0) + "_c.png"
img_c_file = os.path.join("Training_data", "train", img_c_name)

imge_c = io.imread(img_c_file, as_grey=True)
t = imge_c.flatten()
print(imge_c.dtype)
print(t.shape)
print(t)
print("-----------------")
print(imge_c)
print("-------------------")
# look at the image
plt.imshow(imge_c)
plt.show()


data = loadData()
# Plot a random image
trainImg = data[0]
trainLabel = data[1]
sample_number = 0
sample_img = trainImg[sample_number,:].reshape(2,128,118)
print("sample_img shape:",sample_img.shape)
print("-----------------")
print(sample_img[0,:,:])
print("-------------------")
plt.imshow(sample_img[0,:,:], cmap="gray_r")
plt.show()
#print(train_c[sample_number,:-1])
plt.axis('off')
print("Image Label: ", trainLabel[sample_number])
'''