# Program to remove info category from data spreadsheets, add PAT# column and remove Image Type and Feature class
# columns, automated to do this for every .tsv file in a given directory
# Will NOT work if there are any already edited files in the path

import pandas as pd
import numpy as np
import os
pathName = "E:/AllGoodFiles/"

# make sure that the Drobo is mounted and findable
# if the path does not exist the program will end early and give an error message

if os.path.exists(pathName) is False:
    print("ERROR: The path does not exist.")
    exit()

patients = []
PatDirs = []
PatDirs = [os.path.join(pathName, name) for name in os.listdir(pathName)
            if os.path.isdir(os.path.join(pathName, name))]
#print(PatDirs)
# Get a list of the identifiers
for directory in PatDirs:
    patients.append(os.path.split(directory)[1])

#print(patients)

tempFile = {}
filesToProcess = []
# use os.walk() to walk through directory and grab files that we're interested in
for root, dirs, files in os.walk(pathName, topdown = True):
    files = [file for file in files if (file.endswith('.tsv') and not(file.startswith('e')))]  # only grab .tsv files (all we need)
    #dirs[:] = [d for d in dirs if d.startswith('PAT')]  # only look in folders that start with PAT?
    for element in files:
        tempFile = {"Path":root, "File":element}
        filesToProcess.append(tempFile)

listLength = len(filesToProcess)
#print(filesToProcess)

# edit tsv file to desired specs
for i in range(listLength):
    fileName = filesToProcess[i]["File"]
    fileNameEdit = 'e' + os.path.splitext(fileName)[0]+'.csv'
    #print(fileNameEdit)
    pathName = filesToProcess[i]["Path"]

    # generate locations for input and output files
    location = pathName + "/" + fileName
    locationE = pathName + "/" + fileNameEdit
    print(locationE)
    # import .tsv file as panda data frame for manipulation
    df = pd.read_csv(location,  index_col = "Feature Class", parse_dates = True, sep = '\t', header = 0)
    #index_col = "Feature Class",
    df = df.drop("info", axis=0)
    df.reset_index(inplace=True)
    # Remove the extraneous Label information
    #print(os.path.splitext(location)[0])
    #create a column with patient number
    tempa,tempb = location.split("PAT",1)
    patientSplit,imageTypeTsv=tempb.split("/",1)
    df['Patient'] = patientSplit
    df['Label'] = df['Label'].astype(str).str[-9:]
    df['Slice'] = pd.to_numeric(df['Label'].astype(str).str[-1:])-2
    df['Feature'] = df['Image type'] + "_" + df['Feature Class'] + "_" + df['Feature Name'] + "_" + df['Label']
    #df['Feature'] =  os.path.splitext(fileName)[0] + "_" + df['Image type'] + "_" + df['Feature Class'] + "_" + df['Feature Name'] + "_" + df['Label']
    #df.sort_values(by=['Label'], inplace=True)
    # Create a new dataframe with only the Feature and Value
    editdf = pd.DataFrame()

    editdf = df[['Patient','Slice','Feature', 'Value']]
    # replace NAN and INF with 0 in values column, use DataFrame.fillna()
    editdf.replace('inf', 0, inplace = True)  # for some reason inf is not registering as a numpy expression
    editdf.fillna(0, inplace = True)  # inplace = True overwrites the original file, same as df = df.fillna()
    # Save edited data frame to a new tsv file--ends up comma delineated, ok?
    editdf.to_csv(locationE,index=False)
