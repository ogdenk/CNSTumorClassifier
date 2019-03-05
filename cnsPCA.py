import os
import pandas as pd
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import seaborn
from sklearn.ensemble import ExtraTreesClassifier
seaborn.set(style='ticks')
# Wilcoxon signed-rank test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import wilcoxon
from scipy.stats import kruskal

from scipy.stats import mannwhitneyu
#import eli5
#from eli5.sklearn import PermutationImportance
#from sklearn.ensemble import ExtraTreesClassifier


rootDir = 'X:/'
#filename = 'rawavg_nogad1.5.csv'
#filename = 'dataSetEmbSorted.0.1.csv'
#filename = 'dataSetEmb1.30.19.csv'
#filename = 'dataSetGBMwithReRunRadiomics1.csv'
filename='dataSetEmb2.4.19noDataAugFeatureAvg.csv'
datafile = os.path.join(rootDir, filename)
pre_data_df = pd.read_csv(datafile)

slices = 1

total_p_values = []
feature_number = []
#embryonal_data, non_embryonal_data = np.split(pre_data, [rowNumber])  #split these when doing kruskal p values

##the following code prepares the data row and column lengths for the network
  #slices set as 1 for seizure study.  use 2 for no data augmentation tumors. 4 for data augmentation tumors
total_rows=len(pre_data_df.axes[0])
total_columns=len(pre_data_df.axes[1])
len_group = slices**slices
number_of_patients = total_rows/len_group
valFrac = 1-(number_of_patients-1)/number_of_patients #Training fraction all but one patient
trainFrac = 1-valFrac #Validation one patient
total_Val_Acc = []
total_Val_Loss = []
test_patients = []

##the following code pre normalizes the data
data_df=pre_data_df
scaling_data = data_df.as_matrix()[:, 3:]
scaler = StandardScaler()
scaler.fit(scaling_data)
normalized_scaling_data = scaler.transform(scaling_data)  #this array does not contain patient number , patient, tumor type
short_data = pre_data_df[pre_data_df.columns[0:3]]
normalized_data_df = pd.DataFrame(normalized_scaling_data)
short_data_df = pd.DataFrame(short_data)
data_df = pd.concat([short_data_df, normalized_data_df], axis=1)
data_df.set_index('Patient_Number', inplace=True, drop=True)
pre_data_df=data_df

for i in range(int(number_of_patients)):

    data_df=pre_data_df
    testRows=slices**slices
    #testRows = 256
    #testRows = int(valFrac*data.shape[0]+1)  #addd 1 due to computer rounding error in some data sets.  use above 256 if this is happening

    test_data_df, train_data_df = np.split(data_df,[testRows])

    train_data_size = len(train_data_df)
    test_data_size = len(test_data_df)

    train_labels = train_data_df['Tumor_Type'].as_matrix().astype('float32')
    test_labels = test_data_df['Tumor_Type'].as_matrix().astype('float32')
    #train_labels = train_data_df['SzBd400'].as_matrix().astype('float32')
    #test_labels = test_data_df['SzBd400'].as_matrix().astype('float32')

    train_data = train_data_df.as_matrix()[:,2:]
    test_data = test_data_df.as_matrix()[:,2:]


    #this code builds and runs the sequential model. this is used in CNS classifier and leave one out
    model = models.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.5))

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
    print(test_data_df.index[0])
    #print(test_data_df.index[1])
    print(last_val_acc_value)
    print(last_val_loss_value)
    test_patients.append(test_data_df.index[0])
    total_Val_Acc.append(last_val_acc_value)
    total_Val_Loss.append(last_val_loss_value)

    last99Patient_cycle_data = pre_data_df[total_rows - (total_rows-len_group):]
    first1percent_cycle_data = pre_data_df[:total_rows - (total_rows-len_group)]
    pre_data_df = pd.concat([last99Patient_cycle_data, first1percent_cycle_data])

    #print(model.feature_importances_)
print(np.mean(total_Val_Acc))
print(np.mean(total_Val_Loss))
print("validation accuracy = " + str(total_Val_Acc))
print("validation loss = " + str(total_Val_Loss))
print(test_patients)


plt.plot(np.arange(len(total_Val_Acc)), total_Val_Acc, 'bo', label='Validation Acc')
plt.title('Validation Accuracy All Patients')
plt.xlabel('Patient')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(np.arange(len(total_Val_Loss)), total_Val_Loss, 'bo', label='Validation Loss')
plt.title('Loss All Patients')
plt.xlabel('Patient')
plt.ylabel('Loss')
plt.legend()
plt.show()


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


'''
#the following block of code produces the 2 dimensional PCA analysis scatter plot
data = pre_data
scaling_data = data.as_matrix()[:, 3:]
scaler = StandardScaler()
scaler.fit(scaling_data)
data = scaler.transform(scaling_data)
pca = PCA(n_components=2)
pca.fit(data)
x_pca = pca.transform(data)
short_data = pre_data[pre_data.columns[1:3]]
data = pd.DataFrame(data)
short_data = pd.DataFrame(short_data)
x_pca = pd.DataFrame(x_pca)
x_pca_df = pd.concat([short_data, x_pca], axis=1)
fg = seaborn.FacetGrid(data=x_pca_df, hue='Tumor_Type', hue_order=[0, 1], aspect=1.61)
fg.map(plt.scatter, 0, 1).add_legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
df_comp = pd.DataFrame(pca.components_)
plt.figure(figsize=(12,6))
seaborn.heatmap(df_comp,cmap='plasma',)
'''


 # Plot the feature importances of the forest. this works, but the graph is overwhelming
'''
    model = ExtraTreesClassifier()
    model.fit(train_data, train_labels)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(train_data.shape[1]), importances[indices], color="r", xerr=std[indices], align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(train_data.shape[1]), indices)
    plt.ylim([-1, train_data.shape[1]])
'''


'''
    #this code runs the eli5 package for feature importance for sequential model.  unable to install eli5 module 
    def base_model():
        model = Sequential()
        model.add(layers.Dense(8, activation='relu', input_shape=(train_data.shape[1],)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        return model
    X = train_data
    y = train_labels
    my_model = KerasRegressor(build_fn=base_model, **sk_params)
    my_model.fit(X,y)
    perm = PermutationImportance(my_model, random_state=1).fit(X,y)
    eli5.show_weights(perm, feature_names = X.columns.tolist())
'''



'''
##this code runs a Wilcoxon feature analysis for 100 and 80 randomly generated numbers
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(80) + 58
data2 = 5 * randn(100) + 58
# compare samples
stat, p = kruskal(data1, data2)  #substitute in wilcoxon if equal sample sizes
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')
'''


'''
##the following code evaluates the Wilcoxon/kruskal test for each feature value
total_p_values = []
feature_number = []
rowNumber=13056
embryonal_data, non_embryonal_data = np.split(pre_data, [rowNumber])
for x in range((3364-3)):    #go until number of features - 3 to account for patient number, pat, tumor type columns
    data1 = embryonal_data.as_matrix()[:,(x+3):(x+4)]
    data2 = non_embryonal_data.as_matrix()[:,(x+3):(x+4)]
    stat, p = kruskal(data1, data2)
    total_p_values.append(p)
    feature_number.append(x)
    print (p)
    print (x)
print (total_p_values)
print (feature_number)
'''


'''
    model = ExtraTreesClassifier()
    model.fit(train_data, train_labels)
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)
    print()
    print ((train_data.shape[1]))
    sorted_all_importance = importances[indices]
    print (sorted_all_importance)
    print(indices)
    shortened_sorted_all_importance = np.delete(sorted_all_importance,np.s_[5:],axis=0)
    shortened_indices = np.delete(indices,np.s_[5:],axis=0)
    print(shortened_sorted_all_importance)
    print(shortened_indices)
    plt.figure()
    plt.title("Feature importances")
    plt.barh(5, shortened_sorted_all_importance, color="r", xerr=std[shortened_indices], align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(5, shortened_indices)
    plt.ylim([-1, 5])
'''