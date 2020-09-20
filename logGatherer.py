# ***********************
# *****| LIBRARIES |*****
# ***********************
import pandas as pd
import numpy as np
import os
import json

# *****************
# *** GET FILES ***
# *****************
def getFiles( driverPath, directory, basename, extension):  # Define a function that will return a list of files
    pathList = []                                           # Declare an empty array
    directory = os.path.join( driverPath, directory)        # 
    
    for root, dirs, files in os.walk( directory):           # Iterate through roots, dirs and files recursively
        for file in files:                                  # For every file in files
            if os.path.basename(root) == basename:          # If the parent directory of the current file is equal with the parameter
                if file.endswith('.%s' % (extension)):      # If the searched file ends in the parameter
                    path = os.path.join(root, file)         # Join together the root path and file name
                    pathList.append(path)                   # Append the new path to the list
    return pathList  

# ***********************************
# *** GET THE PATHS FOR THE FILES ***
# ***********************************

# Path to the content of the Google Drive 
driverPath = ""

# Sub-directories in the driver
paths = [
    "processed/depression/submission",
    "processed/depression/comment", 
    "processed/AskReddit/submission", 
    "processed/AskReddit/comment"]

files = [None] * len(paths)
for i in range(len(paths)):
  files[i] = getFiles( driverPath, paths[i], "log", "json")
  print("Gathered %i files from %s." % ( len(files[i]), paths[i]))

# ******************************
# *** OPEN AND READ THE LOGS ***
# ******************************
no_folders = len(files)                                 # Get the number of folders, almost always 4 for us
print("Number of data folders: %i.\n" % ( no_folders))  # Print the given message

template_logs = {"removed": 0,                               # Establish a template compatible with the JSONs retrieved
                 "deleted": 0, 
                 "empty": 0, 
                 "error": 0, 
                 "sticky": 0, 
                 "unusable": 0, 
                 "usable": 0}

template_processed = {
        "empty": 0,
        "links": 0,
        "negatives": 0,
        "newlines": 0,
        "symbols": 0,
        "at": 0,
        "numbers": 0,
        "punctuation&numbers": 0, 
        "contractionsPos": 0, 
        "contractionsNeg": 0, 
        "words": 0,
        "wordsAvg": 0,
        "wordLengthAvg": 0,
        "characters": 0,
        "charactersAvg": 0    
}


total = {}                                              # Define a new empty dictionary named "total"
total.update(template_processed)                                  # .update() will change the format of "total" to that of "template"
errors = 0                                              # Total amount of errors caught

# Iterate through all the types of data we gathered, almost always 4
for folder in range(no_folders):
    for file in files[folder]:
        with open(file) as f:
            try:
                object = json.loads( f.read())
            except:
                errors += 1
            for key, value in object.items():
                total[key] += object[key]
    total["wordsAvg"] = total["wordsAvg"] / len(files[folder])
    total["charactersAvg"] = total["charactersAvg"] / len(files[folder])
    total["wordLengthAvg"] = total["wordLengthAvg"] / len(files[folder])
    print("Print the returned logs for %s:\n%s\n" % (paths[folder], total))
    total.update(template_processed)
print("Total number of errors that occured during processing: %i." % (errors))
    