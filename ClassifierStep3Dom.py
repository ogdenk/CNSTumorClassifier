import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import models
from keras import layers
import matplotlib.pyplot as plt

slices=4  #slices set as 1 for seizure study
trainFrac = 0.8065 #Training fraction (includes validation)
valFrac = 0.1935 #Fraction of training data used for validation

rootDir = 'E:/AllGoodFiles/'
#filename = 'rawavg_nogad1.5.csv'
filename = 'testfile.csv'
datafile = os.path.join(rootDir, filename)
#pre_data = pd.read_csv(datafile)
#data = pd.read_csv(datafile,index_col='Patient Number')   #this is the good one for non shuffling
pre_data = pd.read_csv(datafile,index_col='Patient Number')

##the following block of code creates a dataframe with random numbers to test for 50% accuracy

#data2 = pd.DataFrame(np.random.randint(0,1000,size=(20000,3200)),)
#data2.insert(0,'Patient Number',range(0,0+len(data2)))
#data2.insert(1,'Tumor Type',np.random.randint(0,2,size=(len(data2),1)))
#len_group=10
#index_list = np.array(data2.index)
#np.random.shuffle(np.reshape(index_list, (-1, len_group)))
#data2 = data2.loc[index_list, :]


##the following block of code randomizes/shuffles the list of patient data in blocks of 256
len_group = slices**slices
index_list = np.array(pre_data.index)
np.random.shuffle(np.reshape(index_list, (-1, len_group)))
data=pre_data
data = data.loc[index_list, :]

#data = shuffle(pre_data)
#data2=pd.DataFrame(data,index=index_list)
#data3=data.loc['Tumor Type', 'Patient Number']



trainRows = int(trainFrac*data.shape[0])
train_data_df, test_data_df = np.split(data,[trainRows])
train_data_size = len(train_data_df)
test_data_size = len(test_data_df)
train_labels = train_data_df['Tumor Type'].as_matrix().astype('float32')
test_labels = test_data_df['Tumor Type'].as_matrix().astype('float32')
#train_labels = train_data_df['SzBd400'].as_matrix().astype('float32')
#test_labels = test_data_df['SzBd400'].as_matrix().astype('float32')
train_data = train_data_df.as_matrix()[:,2:]  #the 2: ensures that the patient number and tumor type are excluded from the training data.
test_data = test_data_df.as_matrix()[:,2:]

means = train_data.mean(axis=0)
sigmas = train_data.std(axis=0)

train_data = (train_data-means)/sigmas
test_data = (test_data-means)/sigmas

model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(8, activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_data,train_labels, epochs=20, batch_size=8, validation_data=(test_data, test_labels))
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)


acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
acc_values = history_dict['acc']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
