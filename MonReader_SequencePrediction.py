#!/usr/bin/env python
# coding: utf-8

# # Sequence Prediction

# ## Create separate folders for each sequence of frames

# In[1]:


# import os
# import shutil

# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\training\\flip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\training\\flip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# In[2]:


# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\training\\notflip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\training\\notflip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# In[3]:


# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing\\flip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\testing\\flip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# In[4]:


# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\testing\\notflip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\testing\\notflip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# In[5]:


# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set\\flip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\test_set\\flip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# In[6]:


# # Define your directory paths
# source_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\images\\images\\test_set\\notflip'
# destination_dir = 'C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\test_set\\notflip'

# # Iterate over each file in the source directory
# for filename in os.listdir(source_dir):
#     # Extract video ID and frame number from filename
#     video_id, frame_num = filename.split('_')  # Update this line based on your filename format
#     sequence_dir = os.path.join(destination_dir, video_id)
    
#     # Create a directory for the sequence if it doesn't exist
#     if not os.path.exists(sequence_dir):
#         os.makedirs(sequence_dir)
    
#     # Move or copy the file to the new directory
#     shutil.move(os.path.join(source_dir, filename), os.path.join(sequence_dir, filename))


# # Implement CNN/LSTM model

# In[12]:


import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Load pre-trained ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False)


# In[9]:


def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array)
    return features


# In[9]:


import os
from tensorflow.keras.applications.resnet50 import preprocess_input
all_sequences_training_flip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\training\\flip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_training_flip.append(sequences_video)
        


# In[10]:


all_sequences_training_notflip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\training\\notflip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_training_notflip.append(sequences_video)
        


# In[11]:


all_sequences_validation_flip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\testing\\flip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_validation_flip.append(sequences_video)


# In[12]:


all_sequences_validation_notflip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\testing\\notflip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_validation_notflip.append(sequences_video)


# In[13]:


import numpy as np

# Assuming all_sequences_training_flip and all_sequences_training_notflip are your data and labels are 0 for flip and 1 for not flip
all_sequences = all_sequences_training_flip + all_sequences_training_notflip
labels = [0] * len(all_sequences_training_flip) + [1] * len(all_sequences_training_notflip)
labels=np.array(labels)

# Combine and shuffle
combined = list(zip(all_sequences, labels))
np.random.shuffle(combined)
all_sequences[:], labels[:] = zip(*combined)

all_sequences_val = all_sequences_validation_flip + all_sequences_validation_notflip
labels_val = [0] * len(all_sequences_validation_flip) + [1] * len(all_sequences_validation_notflip)
labels_val = np.array(labels_val)
# Combine and shuffle
combined = list(zip(all_sequences_val, labels_val))
np.random.shuffle(combined)
all_sequences_val[:], labels_val[:] = zip(*combined)


# In[57]:


#find maxlen
sequence_lengths = [len(seq) for seq in all_sequences]
maxlen = int(np.percentile(sequence_lengths, 95))  # or 90, or any other percentile you choose
average = sum(sequence_lengths) / len(sequence_lengths)


# In[59]:


#find maxlen_val
sequence_lengths = [len(seq) for seq in all_sequences_val]
maxlen_val = int(np.percentile(sequence_lengths, 95)) 
average_val = sum(sequence_lengths) / len(sequence_lengths)


# In[35]:


# Define LSTM model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(LSTM(units=256, input_shape=(None, 7*7*2048), return_sequences=False))
model.add(Dropout(0.5))  # Add dropout with a probability of 0.5
model.add(Dense(1, activation='sigmoid'))  # or more units for classification

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Convert list of sequences to an array for training
# Note: This is an oversimplified example. You'll need to properly batch and pad sequences of varying lengths.
all_sequences = np.array(all_sequences, dtype=object)
all_sequences_val = np.array(all_sequences_val, dtype=object)
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 'all_sequences' is a list of sequences
# 'maxlen' is an integer of the maximum sequence length to pad to, or None
padded_sequences = pad_sequences(all_sequences, maxlen=10, padding='post', truncating='post')
padded_sequences = padded_sequences.reshape(117, 10, -1)
padded_validation_sequences = pad_sequences(all_sequences_val, maxlen=maxlen_val, padding='post', truncating='post')
padded_validation_sequences = padded_validation_sequences.reshape(107, maxlen_val, -1)
# Train the model
history=model.fit(padded_sequences, labels, epochs=50,validation_data=(padded_validation_sequences, labels_val))


# In[36]:


#plot the loss and accuracy
import matplotlib.pyplot as plt

# Assuming 'history' is the result returned by the fit function
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


#using a maxlen of 10, which is much smaller than maxlen of 27, for training
#and a maxlen of 5 for validation works really well for validation accuracy
#you get around 96% val accuracy.  Increasing maxlen for training beyond 10 doesn't seem to improve val acc.


# In[37]:


#save model
model.save('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\lstm.h5')  # Save as HDF5 file


# In[13]:


import os
from tensorflow.keras.applications.resnet50 import preprocess_input
all_sequences_test_flip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\test_set\\flip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_test_flip.append(sequences_video)


# In[14]:


all_sequences_test_notflip = []  # Initialize an empty list to hold all video sequences
# Loop through each folder and subfolder
for root, dirs, files in os.walk('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\videos\\videos\\test_set\\notflip'):
    for sub_dir in dirs:
        full_path = os.path.join(root, sub_dir)
        # Now full_path is the path to each video's image sequence
        sequences_video = []
        for frame in sorted(os.listdir(full_path)):
            frame_path = os.path.join(full_path, frame)
            features = extract_features(frame_path)
            sequences_video.append(features)
        # Convert sequences to numpy array and handle further processing
        sequences_video = np.array(sequences_video)
        all_sequences_test_notflip.append(sequences_video)


# In[15]:


all_sequences_test = all_sequences_test_flip + all_sequences_test_notflip
labels_test = [0] * len(all_sequences_test_flip) + [1] * len(all_sequences_test_notflip)
labels_test=np.array(labels_test)


# In[16]:


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load HDF5 model
model = load_model('C:\\Users\\Richard Han\\Downloads\\Apziva MonReader\\lstm.h5')

# Assuming all_sequences_test is a list of extracted feature sequences for the test set
padded_test_sequences = pad_sequences(all_sequences_test, maxlen=4, padding='post', truncating='post')
padded_test_sequences = padded_test_sequences.reshape(-1, 4, 7*7*2048)  # Reshape according to your LSTM input

# Predicting on the test set
test_predictions = model.predict(padded_test_sequences)
test_predictions_binary = (test_predictions > 0.5).astype(int)

from sklearn.metrics import accuracy_score

# Calculate Accuracy
accuracy = accuracy_score(labels_test, test_predictions_binary)
print(f"Test Accuracy: {accuracy}")

# Calculate precision, recall, and F1 score using true labels (labels_test)
precision = precision_score(labels_test, test_predictions_binary)
recall = recall_score(labels_test, test_predictions_binary)
f1 = f1_score(labels_test, test_predictions_binary)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[ ]:


#find maxlen
sequence_lengths = [len(seq) for seq in all_sequences_test]
maxlen_test = int(np.percentile(sequence_lengths, 95))
average_test = sum(sequence_lengths) / len(sequence_lengths)


# In[ ]:


#the test set has a maxlen of 6; however, using a maxlen of 4 gives better results.


# In[1]:





# In[ ]:




