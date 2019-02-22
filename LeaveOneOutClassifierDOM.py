import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras import models
from keras import layers
import matplotlib.pyplot as plt

slices=1  #slices set as 1 for seizure study


rootDir = 'x:/'
#filename = 'rawavg_nogad1.5.csv'
filename = 'dataSetEmb2.4.19noDataAugFeatureAvg.csv'
datafile = os.path.join(rootDir, filename)
pre_data = pd.read_csv(datafile,index_col='Patient Number')



total_rows=len(pre_data.axes[0])
total_columns=len(pre_data.axes[1])
len_group = slices**slices
number_of_patients = total_rows/len_group
trainFrac = (number_of_patients-1)/number_of_patients #Training fraction all but one patient
valFrac = 1-trainFrac #Validation one patient
total_Val_Acc = []
total_Val_Loss = []

for i in range(int(number_of_patients)):
    #last256_cycle_data = pre_data.as_matrix()[(total_rows-len_group):,:]

    #index_list = np.array(pre_data.index)
    #np.reshape(index_list, (-1, len_group))
    data=pre_data

    #data2=pd.DataFrame(data,index=index_list)
    #data3=data.loc['Tumor Type', 'Patient Number']
    #data = data.loc[index_list, :]
    #data = shuffle(pre_data)

    trainRows = int(trainFrac*data.shape[0])

    train_data_df, test_data_df = np.split(data,[trainRows])

    train_data_size = len(train_data_df)
    test_data_size = len(test_data_df)

    train_labels = train_data_df['Tumor Type'].as_matrix().astype('float32')
    test_labels = test_data_df['Tumor Type'].as_matrix().astype('float32')
    #train_labels = train_data_df['SzBd400'].as_matrix().astype('float32')
    #test_labels = test_data_df['SzBd400'].as_matrix().astype('float32')

    train_data = train_data_df.as_matrix()[:,3:]
    test_data = test_data_df.as_matrix()[:,3:]

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

    history = model.fit(train_data,train_labels, epochs=20, batch_size=6, validation_data=(test_data, test_labels))
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)


    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    acc_values = history_dict['acc']
    last_val_acc_value = val_acc_values[-1]
    last_val_loss_value = val_loss_values[-1]

    total_Val_Acc.append(last_val_acc_value)
    total_Val_Loss.append(last_val_loss_value)

    lastPatient_cycle_data = pre_data[total_rows - 256:]
    first99percent_cycle_data = pre_data[:total_rows - 256]
    pre_data = pd.concat([lastPatient_cycle_data, first99percent_cycle_data])
print(np.mean(total_Val_Acc))
print(np.mean(total_Val_Loss))
print(total_Val_Acc)
print(total_Val_Loss)
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
