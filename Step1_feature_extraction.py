
# In[1]:
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense, Reshape, Input, InputLayer
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD  # from tensorflow.keras.optimizers import Adam, sgd
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
import random
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, LSTM  # Import GlobalAveragePooling2D
#import imgaug as ia
#import imgaug.augmenters as iaa
import tensorflow as tf
# # Ensure eager execution is enabled
# tf.config.run_functions_eagerly(True)
import keras
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Import GlobalAveragePooling2D
# from keras_efficientnets import ResNet50,EfficientNetB3
from tensorflow.keras.applications import ResNet50
from skimage.util import random_noise
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.color import rgb2yiq, rgb2hed
from sklearn.metrics import classification_report, accuracy_score, f1_score
# tf.compat.v1.disable_eager_execution()
# In[2]:
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay




def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


# set DL parameters
batch_size = 16 # 32*3
# INIT_LR = 1e-3
EPOCHS = 100
EPOCHS_1 = 100 # training B0 each model for example only red images
# num_snapshots_chosed=200
# num_shift_pixels=10
num_balance_for_train_212223=int(582)#min 260*0.8
um_balance_for_test_212223=int(146)#min 260*0.2
num_start_layer = 18 # ou 18
# createFolder("./testvelocity/") #########for instance
createFolder(
    "./test3_BEIT_DFetAFFetSDC_finT_NoDecalage_5clas_train21-23_DF/")  #########for instance

# Define the root input and output directories
input_root_dir = "./test2/"
output_root_dir = "./test3_BEIT_DFetAFFetSDC_finT_NoDecalage_5clas_train21-23_DF/"

image_shape1 = 224
image_shape2 = 224

#######################################################level 1 donnees######################################################

###############RGB
# Save random multiple arrays into a single .npz file
Dir_x_train_RGB = os.path.join(output_root_dir, "x_train_RGB.npz")
Dir_x_test_RGB = os.path.join(output_root_dir, "x_test_RGB.npz")
# Dir_x_test_2023_RGB = os.path.join(output_root_dir, "x_test_2023_RGB.npz")
Dir_x_test_2024_RGB = os.path.join(output_root_dir, "x_test_2024_RGB.npz")

# np.savez(Dir_x_train_RGB, x_train_RGB)
# np.savez(Dir_x_test_RGB, x_test_RGB)
# # np.savez(Dir_x_test_2023_RGB, x_test_2023_RGB)
# np.savez(Dir_x_test_2024_RGB, x_test_2024_RGB)

# Load the random .npz file
x_train_RGB = ( np.load(Dir_x_train_RGB, mmap_mode='r'))["arr_0"]
x_test_RGB = ( np.load(Dir_x_test_RGB, mmap_mode='r'))["arr_0"]
# x_test_2023_RGB = ( np.load(Dir_x_test_2023_RGB, mmap_mode='r'))["arr_0"]
x_test_2024_RGB = ( np.load(Dir_x_test_2024_RGB, mmap_mode='r'))["arr_0"]


###############RGB
# Save random multiple arrays into a single .npz file
Dir_y_train_RGB = os.path.join(output_root_dir, "y_train_RGB.npz")
Dir_y_test_RGB = os.path.join(output_root_dir, "y_test_RGB.npz")
# Dir_y_test_2023_RGB = os.path.join(output_root_dir, "y_test_2023_RGB.npz")
Dir_y_test_2024_RGB = os.path.join(output_root_dir, "y_test_2024_RGB.npz")

# np.savez(Dir_y_train_RGB, y_train_RGB)
# np.savez(Dir_y_test_RGB, y_test_RGB)
# # np.savez(Dir_y_test_2023_RGB, y_test_2023_RGB)
# np.savez(Dir_y_test_2024_RGB, y_test_2024_RGB)

# Load the random .npz file
y_train_RGB = ( np.load(Dir_y_train_RGB, mmap_mode='r'))["arr_0"]
y_test_RGB = ( np.load(Dir_y_test_RGB, mmap_mode='r'))["arr_0"]
# y_test_2023_RGB = ( np.load(Dir_y_test_2023_RGB, mmap_mode='r'))["arr_0"]
y_test_2024_RGB = ( np.load(Dir_y_test_2024_RGB, mmap_mode='r'))["arr_0"]


############################RGB
# Step 1:Initialize the base model with  ResNet50
# base_model = ResNet50(classes=2, input_shape=(image_shape1, image_shape2, 3), weights="None")
weights_path = os.path.join(output_root_dir, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
base_model = ResNet50(include_top=True, input_shape=(image_shape1, image_shape2, 3), weights=weights_path)

# Step 2: Freeze the base model"s layers
base_model.trainable = False


# Step 3: Add custom layers on top of the base model
x = base_model.layers[-2].output   # shape (1000,)
x = Dense(256, activation="relu")(x)
# x = Dropout(0.3)(x)

# x = Dense(64, activation="relu")(x)
# x = Dropout(0.2)(x)

predictions = Dense(6, activation="softmax")(x)

# Step 4: Define the complete model
model = Model(inputs=base_model.input, outputs=predictions)


# Step 5: Train the model with frozen base layers and learning rate
################################set learning rate
def lr_scheduler(epoch):
    if epoch < 10:
        return 0.0001

    if 10 <= epoch < 20:
        return 0.00009

    if 20 <= epoch < 30:
        return 0.00008

    if 30 <= epoch < 40:
        return 0.00007

    if 40 <= epoch < 50:
        return 0.00006

    if 50 <= epoch < 60:
        return 0.00005

    if 60 <= epoch < 70:
        return 0.00004

    if 70 <= epoch < 80:
        return 0.00003
    if 80 <= epoch < 90:
        return 0.00002
    else:
        return 0.00001



lr_schedule = LearningRateScheduler(lr_scheduler)
callbacks_list = [lr_schedule]
# # Reduce learning rate when training loss has stopped improving
# reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.2, # This tells the callback to monitor the training loss.factor=0.2: When the learning rate is reduced, it will be multiplied by this factor. For example, if the current learning rate is 0.01, it will be reduced to 0.002 (0.01 * 0.2) when triggered.
#                               patience=5, min_lr=0.001)#patience=5: The callback will wait for 5 epochs without improvement in the monitored loss before reducing the learning rate. If the loss doesn"t improve for 5 consecutive epochs, the learning rate is reduced.

# Stop training when training loss has stopped improving
early_stop = EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)#patience=50: The callback will allow 10 epochs without improvement before stopping the training. If the loss doesn"t improve for 10 consecutive epochs, training will stop.
#restore_best_weights=True: This parameter ensures that after training stops, the model"s weights are restored to the values that achieved the best training loss during the run. This is useful to ensure that the final model is the best version seen during training, not the one at the end of the training process.

# with tf.device("/GPU:1"):
    # Train the model
# history = model.fit(x=x_train_RGB, y=y_train_RGB,batch_size=batch_size,epochs=1, verbose=1, callbacks=[callbacks_list, early_stop])
# # history = model.fit(x=x_train_RGB, y=y_train_RGB,batch_size=batch_size,epochs=EPOCHS_1, verbose=1, callbacks=[callbacks_list, early_stop])


# Step 6: Unfreeze the base model"s layers for fine-tuning
base_model.trainable = True
# Optionally, freeze some layers and unfreeze only the top N layers, chose the layer for fine tuning
for layer in model.layers[:20]:
    layer.trainable = False

# Step 7: Recompile the model with a lower learning rate
opt = Adam()
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Step 8: Fine-tune the model

# num_samples = x_train_RGB.shape[0]
# num_batches = int(num_samples / batch_size)
# leftover_samples = num_samples % batch_size  # Leftover at the end (less than batch_size)
#
# # for i in range(1):
# for epoch in range(EPOCHS):#for epoch in range(200):
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
#     #  Loop over the batches
#     for i in range(num_batches):
#         history = model.fit(x=x_train_RGB[i * batch_size : (i + 1) * batch_size], y=y_train_RGB[i * batch_size : (i + 1) * batch_size],
#                                              batch_size=batch_size,
#                                              epochs=1, verbose=1, callbacks=[callbacks_list,early_stop])

history = model.fit(x=x_train_RGB,
                          y=y_train_RGB,
                          batch_size=batch_size,
                          epochs= EPOCHS_1,
                          verbose=1,
                          callbacks=[callbacks_list,early_stop])

base_model = []
# ######################################print current time
# #tic=time.clock()
# toc=time.process_time()
# totaltime=toc-tic
# print("totaltime",totaltime)
#
# ############################## loss curve
# #val_loss is the value of cost function for your cross-validation data and loss is the value of cost function for your training data.
# ###############if load history, use history["loss"], if not use history.history["loss"]
# pyplot.plot(history.history["accuracy"], label="train",linewidth=2,color="r")
# # pyplot.plot(history.history["val_accuracy"], label="test",linewidth=2,color="k")
# pyplot.title("Model accuracy",size=20)
# pyplot.ylabel("Accuracy",size=16)
# pyplot.xlabel("Epoch",size=16)
# pyplot.xlim([0,EPOCHS_1])
# pyplot.xticks(range(0, EPOCHS_1 + 1 , 20))
# pyplot.xticks(fontsize=12)
# pyplot.yticks(fontsize=12)
# #pyplot.yscale("log")
# # pyplot.ylim([0,1e-0])
# pyplot.legend(fontsize=14)
# # Create the full directory path including "model"
# RGB_Accuracy_png = os.path.join(output_root_dir, "B0model_withImageNet_RGB_Accuracy_fineTuning.png")
# pyplot.savefig(RGB_Accuracy_png, dpi=150)
# pyplot.show()
#
#
# pyplot.plot(history.history["loss"], label="train",linewidth=2,color="r")
# # pyplot.plot(history.history["val_loss"], label="test",linewidth=2,color="k")
# pyplot.title("Model loss",size=20)
# pyplot.ylabel("Loss",size=16)
# pyplot.xlabel("Epoch",size=16)
# pyplot.xlim([0,EPOCHS_1])
# pyplot.xticks(range(0, EPOCHS_1 + 1 , 20))
# pyplot.xticks(fontsize=12)
# pyplot.yticks(fontsize=12)
# # pyplot.yscale("log")
# pyplot.legend(fontsize=14)
# # Create the full directory path including "model"
# RGB_Loss_png = os.path.join(output_root_dir, "B0model_withImageNet_RGB_Loss_fineTuning.png")
# pyplot.savefig(RGB_Loss_png, dpi=150)
# pyplot.show()

# save model and trining process
RGB_history = os.path.join(output_root_dir, "B0model_withPretrainedImageNet_RGB_history_finT.npy")
np.save(RGB_history,history.history)
# Create the full directory path including "model"
# RGB_model_dir = os.path.join(output_root_dir, "model_jaundice_temoin_champ_bande_RGB_lvl1_2021and2022")
RGB_model_dir = os.path.join(output_root_dir, "model_jaundice_temoin_champ_bande_RGB_lvl1_2021and2022_fineTuning.keras")
# RGB_model_dir_2 = os.path.join(output_root_dir, "model_jaundice_temoin_champ_bande_RGB_lvl1_2021and2022.keras")
# Save the model to a directory
model.save(RGB_model_dir)
# # model.save("./model_jaundice_temoin_champ_bande_RGB_lvl1_2021and2022_fineTuning.keras")
# # tf.saved_model(RGB_model_dir, model)
# model = tf.keras.models.load_model(RGB_model_dir)

# save model and trining process
RGB_history2 = os.path.join(output_root_dir, "B0model_withPretrainedImageNet_RGB_history_2_finT.npy")
np.save(RGB_history2, history)

# In[3]:
# Make predictions and print f1 score
y_test_predictions_ini = model.predict(x_test_RGB) #(469,5)
y_test_predictions = np.argmax(y_test_predictions_ini, axis=1) #469,
# ConRGB y_test if it is one-hot encoded
if len(y_test_RGB.shape) == 2:
    y_test_RGB_new = np.argmax(y_test_RGB, axis=1)
else:
    y_test_RGB_new = y_test_RGB
# ConRGB probabilities to binary predictions using a threshold of 0.5
# y_test_predictions_threshold = (y_test_predictions > 0.5).astype(int)
# Binary classification
f1_RGB = f1_score(y_test_RGB_new, y_test_predictions, average="weighted")
print("f1_RGB", f1_RGB, flush=True)
# # Multi-class classification
# f1 = f1_score(y_test_RGB, y_test_predictions, average="weighted")  # or average="macro"

for i in range(1):
    score_RGB = model.evaluate((x_test_RGB), y_test_RGB, batch_size=batch_size)
    print("score_RGB", score_RGB, flush=True)

# Calculate per-class metrics (accuracy, precision, recall, F1-score)
# classification_report includes precision, recall, f1-score, and support for each class
report = classification_report(y_test_RGB_new, y_test_predictions, target_names=[f"Class {i}" for i in range(5)], digits=5)
print("RGB_Classification Report:", flush=True)
print(report, flush=True)

# F1 score for each class
f1_per_class = f1_score(y_test_RGB_new, y_test_predictions, average=None)  # Calculate F1 score for each class
for i, f1 in enumerate(f1_per_class):
    print(f"F1 score for Class {i}: {f1:.5f}", flush=True)



# Make predictions and print f1 score
y_test_2024_predictions_ini = model.predict(x_test_2024_RGB)
y_test_2024_predictions = np.argmax(y_test_2024_predictions_ini, axis=1)
# ConRGB y_test if it is one-hot encoded
if len(y_test_2024_RGB.shape) == 2:
    y_test_2024_RGB_new = np.argmax(y_test_2024_RGB, axis=1)
else:
    y_test_2024_RGB_new = y_test_2024_RGB
# ConRGB probabilities to binary predictions using a threshold of 0.5
# y_test_2024_predictions_threshold = (y_test_2024_predictions > 0.5).astype(int)
# Binary classification
f1_2024_RGB = f1_score(y_test_2024_RGB_new, y_test_2024_predictions, average="weighted")
print("f1_2024_RGB", f1_2024_RGB, flush=True)


for i in range(1):
    score_2024_RGB = model.evaluate((x_test_2024_RGB), y_test_2024_RGB, batch_size=batch_size)
    print("score_2024_RGB", score_2024_RGB, flush=True)

# Calculate per-class metrics (accuracy, precision, recall, F1-score)
# classification_report includes precision, recall, f1-score, and support for each class
report = classification_report(y_test_2024_RGB_new, y_test_2024_predictions, target_names=[f"Class {i}" for i in range(5)], digits=5)
print("2024RGB_Classification Report:", flush=True)
print(report, flush=True)
# Generate confusion matrix
conf_matrix_RGB = confusion_matrix(y_test_RGB_new, y_test_predictions)
conf_matrix_RGB = np.round(conf_matrix_RGB, 5)
print("conf_matrix_RGB",conf_matrix_RGB, flush=True)


# Generate confusion matrix
conf_matrix_RGB_2024 = confusion_matrix(y_test_2024_RGB_new, y_test_2024_predictions)
conf_matrix_RGB_2024 = np.round(conf_matrix_RGB_2024, 5)
print("conf_matrix_RGB_2024",conf_matrix_RGB_2024, flush=True)
import pandas as pd
# Save the confusion matrix values to a CSV file
conf_matrix_RGB = pd.DataFrame(conf_matrix_RGB, index=[f"Class {i}" for i in range(5)], columns=[f"Class {i}" for i in range(5)])
conf_matrix_RGB.to_csv('confusion_matrix_values_RGB.csv')




# F1 score for each class
f1_per_class = f1_score(y_test_2024_RGB_new, y_test_2024_predictions, average=None)  # Calculate F1 score for each class
for i, f1 in enumerate(f1_per_class):
    print(f"2024 F1 score for Class {i}: {f1:.5f}", flush=True)





################################################################################################################feature extraction
# Save random multiple arrays into a single .npz file
Dir_x_train_RGB = os.path.join(output_root_dir, "x_train_RGB.npz")
Dir_x_test_RGB = os.path.join(output_root_dir, "x_test_RGB.npz")
# Dir_x_test_2023_RGB = os.path.join(output_root_dir, "x_test_2023_RGB.npz")
Dir_x_test_2024_RGB = os.path.join(output_root_dir, "x_test_2024_RGB.npz")

# np.savez(Dir_x_train_RGB, x_train_RGB)
# np.savez(Dir_x_test_RGB, x_test_RGB)
# # np.savez(Dir_x_test_2023_RGB, x_test_2023_RGB)
# np.savez(Dir_x_test_2024_RGB, x_test_2024_RGB)
#
# Load the random .npz file
x_train_RGB = ( np.load(Dir_x_train_RGB, mmap_mode='r'))["arr_0"]
x_test_RGB = ( np.load(Dir_x_test_RGB, mmap_mode='r'))["arr_0"]
# x_test_2023_RGB = ( np.load(Dir_x_test_2023_RGB, mmap_mode='r'))["arr_0"]
x_test_2024_RGB = ( np.load(Dir_x_test_2024_RGB, mmap_mode='r'))["arr_0"]


# Save random multiple arrays into a single .npz file
Dir_y_train_RGB = os.path.join(output_root_dir, "y_train_RGB.npz")
Dir_y_test_RGB = os.path.join(output_root_dir, "y_test_RGB.npz")
# Dir_y_test_2023_RGB = os.path.join(output_root_dir, "y_test_2023_RGB.npz")
Dir_y_test_2024_RGB = os.path.join(output_root_dir, "y_test_2024_RGB.npz")

# np.savez(Dir_y_train_RGB, y_train_RGB)
# np.savez(Dir_y_test_RGB, y_test_RGB)
# # np.savez(Dir_y_test_2023_RGB, y_test_2023_RGB)
# np.savez(Dir_y_test_2024_RGB, y_test_2024_RGB)

# Load the random .npz file
y_train_RGB = ( np.load(Dir_y_train_RGB, mmap_mode='r'))["arr_0"]
y_test_RGB = ( np.load(Dir_y_test_RGB, mmap_mode='r'))["arr_0"]
# y_test_2023_RGB = ( np.load(Dir_y_test_2023_RGB, mmap_mode='r'))["arr_0"]
y_test_2024_RGB = ( np.load(Dir_y_test_2024_RGB, mmap_mode='r'))["arr_0"]
#



# In[6]:
##############################################level 2 deep fusion#########################################
RGB_model_dir = os.path.join(output_root_dir, "model_jaundice_temoin_champ_bande_RGB_lvl1_2021and2022_fineTuning.keras")

model_champ_RGB = keras.models.load_model(RGB_model_dir)




outputs_RGB = [model_champ_RGB.layers[num_start_layer-1].output]  # 56 56 256
model_fusion_RGB = Model(inputs=model_champ_RGB.inputs, outputs=outputs_RGB)



import numpy as np
import os

def process_and_save_in_batches(model, data, output_path, batch_size=32):
    """
    Processes data in batches using the given model and saves each batch to a .npz file incrementally.
    """
    num_batches = len(data) // batch_size + int(len(data) % batch_size != 0)  # Calculate number of batches

    # Create an empty dictionary to store all batches
    all_batches = {}

    for i in range(num_batches):
        batch = data[i * batch_size: (i + 1) * batch_size]  # Get current batch
        feature_maps_batch = model.predict(batch)  # Predict feature maps

        # Add the batch to the dictionary with a unique key
        all_batches[f'feature_maps_batch_{i}'] = feature_maps_batch

        del feature_maps_batch  # Free memory

    # Save the dictionary with all batches to a single .npz file
    np.savez(output_path, **all_batches)

    # Dir_feature_maps_pré_Rededge = os.path.join(output_root_dir, "feature_maps_pré_Rededge_32.npz")
    # feature_Rededge = np.load(Dir_feature_maps_pré_Rededge)


    del data  # Free the original dataset


# Process each dataset with its corresponding model and save results
datasets_models = [
    (model_fusion_RGB, x_train_RGB, "feature_maps_pré_RGB_32.npz"),

    (model_fusion_RGB, x_test_RGB, "feature_maps_test_RGB_32.npz"),


    (model_fusion_RGB, x_test_2024_RGB, "feature_maps_test_2024_RGB_32.npz"),


]

# output_root_dir = "output_feature_maps"

# Ensure output directory exists
os.makedirs(output_root_dir, exist_ok=True)

# Process each dataset in batches and save
for model, dataset, filename in datasets_models:
    if dataset is not None and len(dataset) > 0:
        output_path = os.path.join(output_root_dir, filename)
        process_and_save_in_batches(model, dataset, output_path, batch_size)







