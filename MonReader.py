#!/usr/bin/env python
# coding: utf-8

# # Data preparation

# In[1]:


#suppress tensorflow warnings
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress all warnings

#count number of images

import os

def count_images(directory):
    # Count the number of files in the given directory
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# Set the paths to your flip and nonflip directories
flip_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images_original\\images\\training\\flip'
nonflip_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images_original\\images\\training\\notflip'

# Count images in each directory
num_flip_images = count_images(flip_dir)
num_nonflip_images = count_images(nonflip_dir)

print(f"Number of flip images in training set: {num_flip_images}")
print(f"Number of nonflip images in training set: {num_nonflip_images}")

# Set the paths to your flip and nonflip directories
flip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images_original\\images\\testing\\flip'
nonflip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images_original\\images\\testing\\notflip'

# Count images in each directory
num_flip_images_test = count_images(flip_dir_test)
num_nonflip_images_test = count_images(nonflip_dir_test)

print(f"Number of flip images in testing set: {num_flip_images_test}")
print(f"Number of nonflip images in testing set: {num_nonflip_images_test}")


# In[2]:


# #Split the testing set of images into a validation set and a test set
# import os
# import shutil
# import random
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing'  # Replace with the path to your current 'testing' folder
# test_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set'  # Replace with the path to your new test set folder

# # Categories
# categories = ['flip', 'notflip']

# #create test folder
# if not os.path.exists(test_dir):
#     os.mkdir(test_dir)

# for category in categories:
#     category_dir = os.path.join(test_dir, category)
#     if not os.path.exists(category_dir):
#         os.mkdir(category_dir)

# #split the data
# split_ratio = 0.5  # Adjust as needed

# for category in categories:
#     category_path = os.path.join(source_dir, category)
#     images = os.listdir(category_path)
#     random.shuffle(images)  # Shuffle the images
    
#     # Calculate the split index
#     split_index = int(len(images) * split_ratio)
    
#     # Select images to move
#     images_to_move = images[:split_index]
    
#     # Move selected images to the test folder
#     for image in images_to_move:
#         src_path = os.path.join(category_path, image)
#         dest_path = os.path.join(test_dir, category, image)
#         shutil.move(src_path, dest_path)


# In[3]:


#count how many images in testing set and test_set
# Set the paths to your flip and nonflip directories
flip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing\\flip'
nonflip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing\\notflip'

# Count images in each directory
num_flip_images_test = count_images(flip_dir_test)
num_nonflip_images_test = count_images(nonflip_dir_test)

print(f"Number of flip images in validation set: {num_flip_images_test}")
print(f"Number of nonflip images in validation set: {num_nonflip_images_test}")

# Set the paths to your flip and nonflip directories
flip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set\\flip'
nonflip_dir_test = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set\\notflip'

# Count images in each directory
num_flip_images_test = count_images(flip_dir_test)
num_nonflip_images_test = count_images(nonflip_dir_test)

print(f"Number of flip images in test set: {num_flip_images_test}")
print(f"Number of nonflip images in test set: {num_nonflip_images_test}")


# # Single Image Prediction using Transfer Learning

# ## Resnet model

# In[4]:


# #use ResNet50
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.callbacks import EarlyStopping

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# train_data_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\training' # replace with your path
# test_data_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing'  # replace with your path
# test_set_data_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set'

# #set up data generators
# train_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,  # Preprocesses the data using ResNet50's method
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
# test_datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input  # Apply ResNet50 preprocessing
# )

# # Load images from directory
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(224, 224),  # Ensure this matches ResNet's expected input size
#     batch_size=32,
#     class_mode='binary',  # or 'categorical'
#     shuffle=True
# )

# validation_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(224, 224),  # Ensure this matches ResNet's expected input size
#     batch_size=32,
#     class_mode='binary',  # or 'categorical'
#     shuffle=True
# )

# test_generator = test_datagen.flow_from_directory(
#     test_set_data_dir,  # replace with your test data path
#     target_size=(224, 224),  # or the input size of your model
#     batch_size=1,  # or a batch size that divides your total number of test samples
#     class_mode='binary',  # or 'categorical' for multi-class
#     shuffle=False)


# In[5]:


# #load pretrained model
# base_model = ResNet50(weights='imagenet', include_top=False)


# In[6]:


# #add custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer

# # Add a fully connected layer
# x = Dense(1024, activation='relu')(x)

# # Add a logistic layer for binary classification
# predictions = Dense(1, activation='sigmoid')(x)

# # The model we will train
# model = Model(inputs=base_model.input, outputs=predictions)


# In[7]:


# #freeze layers of pretrained model
# for layer in base_model.layers:
#     layer.trainable = False


# In[8]:


# # Early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)


# In[9]:


# #compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


# #train the model
# history = model.fit(
#     train_generator,
#     epochs=100,
#     callbacks=[early_stopping],
#     validation_data=validation_generator
# )


# In[11]:


# #plot the loss and accuracy
# import matplotlib.pyplot as plt

# # Assuming 'history' is the result returned by the fit function
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


# In[12]:


# #Fine-tuning
# import tensorflow as tf
# # Unfreeze some top layers of the model
# for layer in base_model.layers[-20:]:
#     layer.trainable = True

# # Re-compile the model for fine-tuning
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Continue training
# history_fine = model.fit(
#     train_generator,
#     epochs=5,
#     validation_data=validation_generator
# )


# In[13]:


# #save model
# model.save('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\resnet.h5')  # Save as HDF5 file


# In[14]:


# import numpy as np
# #Evaluate model on test set
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_accuracy}")
# #calculate f1 score
# from sklearn.metrics import precision_score, recall_score, f1_score

# # Predictions
# test_predictions = model.predict(test_generator)
# test_predictions_binary = (test_predictions > 0.5).astype(np.int64)

# # True labels
# true_labels = test_generator.classes

# # Calculate precision, recall, and F1 score
# precision = precision_score(true_labels, test_predictions_binary)
# recall = recall_score(true_labels, test_predictions_binary)
# f1 = f1_score(true_labels, test_predictions_binary)

# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")


# # Apply Saved Models to New Images

# In[15]:


#apply resnet model to a new image
#after training the resnet model in google colab using gpu, I saved the model as resnet2.h5
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load HDF5 model
model = load_model('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\resnet2.h5')

# Load and preprocess the image
img_path = 'sample2_notflip.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)  # preprocess the image using ResNet's preprocess_input

# Predict the class
predictions = model.predict(img_array)


# Predict the class
predictions = model.predict(img_array)
print(predictions)  # or further process predictions as needed


# In[16]:


#apply cnn model to a new image
#after training the cnn model in google colab using gpu, I saved the model as cnn.h5
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load HDF5 model
model = load_model('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\cnn.h5')

# Load and preprocess the image
img_path = 'sample2_notflip.jpg'
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)
# Scale the image
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)


# Predict the class
predictions = model.predict(img_array)


# Predict the class
predictions = model.predict(img_array)
print(predictions)  # or further process predictions as needed


# In[17]:


#The fine-tuned resnet model appeared to perform better than the cnn model on realistic-looking photos of pages of books; 
#however, despite both models having high test set accuracy,
#they both performed poorly when it came to images that were like stock photos and not as realistic-looking
#images of photos of pages of books.  Both models don't seem to generalize well beyond the types of images they were trained on.


# In[ ]:




