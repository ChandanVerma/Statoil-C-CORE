from Deep_NN_prereq import *
from Deep_NN_model import *
import os
from dnn_app_utils_v2 import *
#import keras
#import tensorflow

data_dir = ''
train = pd.read_json("F:/Competitions/Kaggle/Statoil/train.json")
train_y=train['is_iceberg']
test = pd.read_json("F:/Competitions/Kaggle/Statoil/test.json")


def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)


train_x = get_scaled_imgs(train)
train_y = np.array(train['is_iceberg'])


train.inc_angle = train.inc_angle.replace('na',0)
idx_tr = np.where(train.inc_angle>0)


train_y = train_y[idx_tr[0]]
train_x = train_x[idx_tr[0],...]
test.inc_angle = test.inc_angle.replace('na',0)
test_x = (get_scaled_imgs(test))
# parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

train_x_flatten = train_x.reshape(train_x.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
train_y = train_y.reshape(1471, 1)
print ("train_y's shape: " + str(train_y.shape))

layers_dims = [16875, 1604, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)


# my_image = "my_image.jpg" # change this to the name of your image file 
# my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
# ## END CODE HERE ##

# fname = "images/" + my_image
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
# my_predicted_image = predict(my_image, my_label_y, parameters)

# plt.imshow(image)
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")