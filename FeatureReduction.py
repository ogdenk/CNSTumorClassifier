'''
This program is to reduce the number of features in the un-augmented data.
This is a modified version of the step 2 program that augments the data.
'''


import os
import numpy as np
import pandas as pd

pathName = "X:/Test"
augmentation = True

# make sure that the Drobo is mounted and findable
os.getcwd()
# if the path does not exist the program will end early and give an error message
if os.path.exists(pathName) is False:
    print("ERROR: The path does not exist.")
    exit()

# display all tsv files in project folder

listOfFiles = list()
listOfPAT = list()
numberFeatures = 841  # the number of features from radiomics after processed and cleaned using BTDataEditor.py
# listOfFNames = list()

# use os.walk() to walk through directory and grab files that we're interested in
for root, dirs, files in os.walk(pathName, topdown=True):
    files = [file for file in files if file.endswith('.csv')]  # only grab .tsv files (all we need)
    files = [file for file in files if file.startswith('e')]  # only grab edited files
    dirs[:] = [d for d in dirs if d.startswith('PAT')]  # only look in folders that start with PAT
    listOfFiles += [os.path.join(root, file) for file in files]
    # listOfFNames += files  # create list of .tsv files from all PAT folders
    listOfPAT += dirs  # only gives one instance each instead of listing the folder name for each file

# remove pathName and filename from root list, this gives us a list of patient names with one per file
j = 0
listLength = len(listOfFiles)
numberOfPatientsTotal = len(listOfPAT)
while j < listLength:
    listOfFiles[j] = listOfFiles[j].replace(pathName + '/', '')
    listOfFiles[j] = listOfFiles[j][0:8]  # remove the file name by keeping only the first 8 char
    j = j + 1
i = 0

# sort list into alphabetical order to ensure correct assignment of tumor type and data
listOfPAT.sort()
listOfFiles.sort()
# read in the excel file containing patient #, tumor type, and number of slices
# the excel file has been edited to just include relevant data, this is located on sheet 2
excel_df = pd.read_excel(pathName + '/SliceData.xls', sheet_name='Sheet2', header=0)

# use slice number to determine number of columns ie the number of augmented data sets, 4 slices = 256 etc
slices = excel_df['Slice_Num'].tolist()

# import .tsv file as panda data frame for manipulation, use one as a template to generate attribute list
tsv_df = pd.read_csv(pathName + "/" + listOfPAT[0] + "/eFlair.csv", index_col=0, parse_dates=True, sep=',', header=0)
numberFeatures = int(tsv_df.shape[0]/slices[0])

x = 0
for sliceNumber in slices:
    if(augmentation):
        slices[x] = pow(4, sliceNumber)
    else:
        slices[x] = sliceNumber
    x = x + 1
col_num = sum(slices)
dataSet = np.empty([numberFeatures*4, col_num], dtype = object)
total_pats = len(listOfPAT)
# there are 841 different attributes calculated by slicer that we are interested in **multiply b 4 for different files
# use dtype = object to ensure there are no errors including both string and float data
# create an empty data set with 841 rows (# of attributes) and the previously calculated number of columns
# set attribute names to first column
# not in while loop as we only want to do this once at the beginning
info_entries = tsv_df['Feature'].tolist()
attributes = np.array(info_entries)
# trim attributes to the first instance (ie: 841)
dataSet = np.insert(dataSet, 0, attributes, axis=1)  # inserts attributes list as a new column
# at the beginning of the data set

# create a list of patient names, there should be approx 256 entries of each name and set to first row of dataSet
# number of name repeats will depend on number of slices
count = 0
pat = 0
i = 1
patNum = np.empty([1, col_num + 1], dtype=object)  # col_num + 1 as we want an empty space to
# fill with 'Patient Number' later
current = listOfPAT[0]
patNum[0, 0] = ''
while i <= col_num:
    patNum[0, i] = current
    if count == slices[pat]:
        count = 0
        pat = pat + 1
        if pat < total_pats:
            current = listOfPAT[pat]
    i = i + 1
    count = count + 1
dataSet = np.insert(dataSet, 0, patNum, axis=0)  # add patient name row to the data set at the beginning
# create a list of tumor types, 256 entries for each patient then add tumorType row to dataSet matrix
tumor_Type = excel_df['Type'].tolist()
tumorType = np.empty([1, col_num + 1], dtype=object)
i = 1

# use data gathered from the excel file to create a list of tumor types, with the proper number of repeats for each
# patient
counter = 0
tumor = 0
current = tumor_Type[0]
tumorType[0, 0] = ''
while i <= col_num:
    tumorType[0, i] = current
    if counter == slices[tumor]:
        counter = 0
        tumor = tumor + 1
        if tumor < total_pats:
            current = tumor_Type[tumor]
    i = i + 1
    counter = counter + 1
dataSet = np.insert(dataSet, 1, tumorType, axis=0)  # add row just under the patient number row
# set headers for first two rows
dataSet[0, 0] = 'Patient Number'
dataSet[1, 0] = 'Tumor Type'  # 0: Medulloblastoma, 1: Pilocytic Astrocytoma, 2: Ependymoma
patient_Num = 0  # count number of patients completed
while patient_Num < numberOfPatientsTotal:
    patientNum = listOfPAT[patient_Num]  # label # for the current patient, ie: patient_num(0) = PAT00010 etc.
    # generate locations for input file
    location = pathName + "/" + patientNum
    # import .tsv file as panda data frame for manipulation
    # check to make sure all files exist for this patient, as none have all
    print(location)
    if os.path.exists(location + "/eFlair.csv") is True:
        tsvFlair_df = pd.read_csv(location + "/eFlair.csv", index_col=0, parse_dates=True, sep=',', header=0)
    if os.path.exists(location + "/eT1.csv") is True:
        tsvT1_df = pd.read_csv(location + "/eT1.csv", index_col=0, parse_dates=True, sep=',', header=0)
    if os.path.exists(location + "/eT2.csv") is True:
        tsvT2_df = pd.read_csv(location + "/eT2.csv", index_col=0, parse_dates=True, sep=',', header=0)
    if os.path.exists(location + "/eDWI.csv") is True:
        tsvADC_df = pd.read_csv(location + "/eDWI.csv", index_col=0, parse_dates=True, sep=',', header=0)  # set as ADC
        # for convenience's sake
    if os.path.exists(location + "/eADC.csv") is True:
        tsvADC_df = pd.read_csv(location + "/eADC.csv", index_col=0, parse_dates=True, sep=',', header=0)
    # determine number of slices for each patient
    slice_num = 4  # initialize
    # use pandas to find 293-296, if exists add 1 to slice_num: MAX = 4, MIN = 1
#    if tsvT1_df.index.str.contains('293').any():  # check the index of T1df to see if it contains the slice #s anywhere
#        slice_num = slice_num + 1
#        print(slice_num)
#    if tsvT1_df.index.str.contains('294').any():
#        slice_num = slice_num + 1
#    if tsvT1_df.index.str.contains('295').any():
#        slice_num = slice_num + 1
#    if tsvT1_df.index.str.contains('296').any():
#        slice_num = slice_num + 1
#        print(slice_num)


    # create data augmentation vector and initialize counter variables, value holder variables
    i = 0
    j = 0
    k = 0
    m = 0
    row = 0
    column = 0
    T1_value = 0
    T2_value = 0
    Flair_value = 0
    ADC_value = 0
    valueVector = [] * 4  # initialize a vector that will hold 4 values
    column_num = pow(4, slice_num)

    # iterate through the rows
    valuesT1 = tsvT1_df['Value']
    valuesT2 = tsvT2_df['Value']
    valuesFlair = tsvFlair_df['Value']
    valuesADC = tsvADC_df['Value']  # may be DWI, ADC is being used as a catchall for both for simplicity's sake

    # create augmented data by stepping through the slices, start by iterating through ADC then work backwards
    # ie: [1,1,1,1],[1,1,1,2],[1,1,1,3],[1,1,1,4],[1,1,2,1],[1,1,2,2] ... [4,4,4,1], [4,4,4,2], [4,4,4,3], [4,4,4,4]
    while row < numberFeatures:
        # iterate through the columns
        while column < column_num:
            # walk through T1
            while i < slice_num:
                index = i * numberFeatures + row
                T1_value = valuesT1.iloc[index]
                i = i + 1
                # walk through T2
                while j < slice_num:
                    index = j * numberFeatures + row
                    T2_value = valuesT2.iloc[index]
                    j = j + 1
                    # walk through Flair
                    while k < slice_num:
                        index = k * numberFeatures + row
                        Flair_value = valuesFlair.iloc[index]
                        k = k + 1
                        # walk through ADC (or DWI listed as ADC for convenience)
                        while m < slice_num:
                            index = m * numberFeatures + row
                            ADC_value = valuesADC.iloc[index]
                            m = m + 1
                            # determine the rows and columns that the data should be inserted into depending upon
                            # place in the loop and image type ie: T1 data will be between 0 and 840,
                            # T2 between 841 and 1681 etc
                            dataRowT1 = (row + 2)
                            dataRowT2 = (row + 2) + numberFeatures
                            dataRowFlair = (row + 2) + numberFeatures*2
                            dataRowADC = (row + 2) + numberFeatures*3
                            dataColumn = column + 1 + patient_Num * pow(4, slice_num)  # determine location for data
                            dataSet[dataRowT1, dataColumn] = T1_value
                            dataSet[dataRowT2, dataColumn] = T2_value
                            dataSet[dataRowFlair, dataColumn] = Flair_value
                            dataSet[dataRowADC, dataColumn] = ADC_value
                            if column < column_num:
                                column = column + 1
                            if column > (column_num - 1):  # iterate to next row don't just restart
                                i = slice_num
                                j = slice_num
                                k = slice_num
                                m = slice_num
                        m = 0
                    k = 0
                j = 0
            i = 0
        column = 0
        row = row + 1  # move to next attribute
    patient_Num = patient_Num + 1  # move to next patient and start the process over
# convert numpy array to a data frame and then save df as a tsv file
df = pd.DataFrame(dataSet)
dfTranspose = df.T
#pd.DataFrame.to_csv(df, pathName + '/dataSet.tsv', sep=',', header = False, index = False)
pd.DataFrame.to_csv(dfTranspose, pathName + '/dataSetCerNoSort.csv', sep=',', header = False, index = False)
pd.DataFrame.to_hdf(dfTranspose, pathName + '/data.h5', key = 'Tumor_Data', mode='w')
testDF = pd.read_hdf(pathName + '/data.h5', key = 'Tumor_Data', mode = 'r')
print("Done!")
