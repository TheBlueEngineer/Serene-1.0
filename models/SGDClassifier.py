# ***********************
# *****| LIBRARIES |*****
# ***********************
import pandas as pd
import numpy as np
import os
import json
from stopwords import stopwords_partial
import scipy.sparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix

# ******************************
# *****| GLOBAL VARIABLES |*****
# ******************************
test_size = 0.2
rand_state_splitter = 1000
rand_state_nn = 1000
max_iter = 300
sentence_limiter = 1000

max_files_depr_sub, max_files_depr_comm = 5,0                          # Max files you want to get from Depression category
max_files_ask_sub, max_files_ask_comm = 5,0                            # Max files you want t0 get from AskReddit category

paths = ["processed/depression/submission",
         "processed/depression/comment", 
         "processed/AskReddit/submission", 
         "processed/AskReddit/comment"]
# *******************************
# *****| UTILITY FUNCTIONS |*****
# *******************************
# *****************
# *** GET FILES ***
# *****************
def getFiles( directory, basename, extension):              # Define a function that will return a list of files
    pathList = []                                           # Declare an empty array
    for root, dirs, files in os.walk( directory):           # Iterate through roots, dirs and files recursively
        for file in files:                                  # For every file in files
            if os.path.basename(root) == basename:          # If the parent directory of the current file is equal with the parameter
                if file.endswith('.%s' % (extension)):      # If the searched file ends in the parameter
                    path = os.path.join(root, file)         # Join together the root path and file name
                    pathList.append(path)                   # Append the new path to the list
    return pathList  

# ****************************************
# *** GET DATA INTO A PANDAS DATAFRAME ***
# ****************************************
def getDataFrame( listFiles, maxFiles):
    pd_list = []
    df_final = []
    for file in listFiles:
        with open(file) as f:
            if maxFiles == 0:
                break
            else:
                maxFiles -= 1
            parsedData = json.loads( f.read())                  # Get the data from the JSON file
            objects = parsedData['data']                        # Get objects stored in the 'data'
            pd_df = pd.DataFrame(objects)                       # Add the objects to a Pandas DataFrame
            pd_list.append(pd_df)                               # Append the DataFrame to a list of DataFrames
    if(len(pd_list) > 0):
        df_final = pd.concat(pd_list)
    return df_final

# **********************
# ***| TEXT LIMITER |***
# **********************
def textLimiter( fileList, limit):
    newList = []
    itteration = 0
    for sentence in fileList:
        itteration += 1
        print("Text limiter sentence: (%i/%i)." % ( itteration, len(fileList)))
        if len(sentence) > limit:
            sentences = sentence[:limit]
        else:
            sentences = sentence.ljust(limit)
        newList.append(sentences)
    return newList

# ***************************
# *****| MAIN FUNCTION |*****
# ***************************
# ************************
# *** PROCESS THE DATA ***
# ************************
if(1):
    print("Step 1: Gather the data from the files.")
    files_depr_sub = getFiles(paths[0], "text", "json")
    print("Gathered %i files from %s." % ( len(files_depr_sub), paths[0]))
    files_depr_comm = getFiles(paths[1], "text", "json")
    print("Gathered %i files from %s." % ( len(files_depr_comm), paths[1]))
    files_ask_sub = getFiles(paths[2], "text", "json")
    print("Gathered %i files from %s." % ( len(files_ask_sub), paths[2]))
    files_ask_comm = getFiles(paths[3], "text", "json")
    print("Gathered %i files from %s." % ( len(files_ask_comm), paths[3]))
    print("Step 1: Completed.\n")
    
    # Get the pandas data frames for each category
    print("Step 2: Build the Pandas DataFrames for each category.")
    pd_depr_sub = getDataFrame( files_depr_sub, max_files_depr_sub)
    pd_ask_sub = getDataFrame( files_ask_sub, max_files_ask_sub)
    pd_depr_comm = getDataFrame( files_depr_comm, max_files_depr_comm)
    pd_ask_comm = getDataFrame( files_ask_comm, max_files_ask_comm)
    
    # Get the length of each data frame
    no_depr_sub = len( pd_depr_sub)
    no_depr_comm = len( pd_depr_comm)
    no_ask_sub = len( pd_ask_sub)
    no_ask_comm = len( pd_ask_comm)
    
    # PRINT information regarding the dataframes just built
    print("Built dataframe with data from depression submissions: %i objects." % ( no_depr_sub))
    print("Built dataframe with data from depression comments: %i objects." % ( no_depr_comm))
    print("Built dataframe with data from AskReddit submissions: %i objects." % ( no_ask_sub))
    print("Built dataframe with data from AskReddit comments: %i objects." % ( no_ask_comm))
    print("Step 2: Completed.\n")
    
    # Let's rest the list of paths so that we can free some memory, in order to reduce the memory cost of the program
    files_depr_sub, files_depr_comm, files_ask_sub, files_ask_comm = None, None, None, None
    
    # Unite all the above data frames
    print("Step 3: Concatenate all the Pandas DataFrames into one.")
    pd_final, pd_final_temporary = [], []
    if no_depr_sub != 0:
        pd_final_temporary.append( pd_depr_sub)
    
    if no_depr_comm != 0:
        pd_final_temporary.append( pd_depr_comm)
    
    if no_ask_sub != 0:
        pd_final_temporary.append( pd_ask_sub)
    
    if no_ask_comm != 0:
        pd_final_temporary.append( pd_ask_comm)
    # Get the final pandas data frame with all the data needed
    pd_final = pd.concat(pd_final_temporary)
    pd_depr_sub, pd_depr_comm, pd_ask_sub, pd_ask_comm = None, None, None, None
    print("Step 3: Completed.\n")
    
    text = pd_final['text'].values
    label = pd_final['label'].values
    
    # text = textLimiter(text, sentence_limiter)
    
    # Split the data into training and testing data
    print("Step x: Split the data.")
    sentences_train, sentences_test, y_train, y_test = train_test_split(text, 
                                                                        label, 
                                                                        test_size = test_size, 
                                                                        shuffle = True, 
                                                                        random_state = rand_state_splitter)
    print("Step x: Completed.")
    # Implement a TFIDF Vectorizer for words
    print("Step x: Define the TFIDF Vectorizer and fit the data.")

    vectorizer = TfidfVectorizer(analyzer='word',
                                 min_df = 0.1,
                                 norm='l2',
                                 ngram_range = (1,2),
                                 stop_words = 'english',
                                 smooth_idf = True
                                 )

    # Get the vocabulary from the sentences_train
    vectorizer_fit = vectorizer.fit(sentences_train)
    delattr(vectorizer_fit, 'stop_words_')
    file = "salut.txt"
    with open(file, "w") as f:
        f.write( str(vectorizer_fit.vocabulary_))
    
    # Transform the data for training and testing into a sparse matrix
    x_train = vectorizer_fit.transform(sentences_train)
    x_test = vectorizer_fit.transform(sentences_test)
    print("Step x: Completed.")
    sys.exit(0)
    print("Step x: Trying to save the vector as a sparse matrix.")
    scipy.sparse.save_npz('training_data_%i_%i.npz' % ( max_files_depr_sub, max_files_ask_sub), x_train)
    scipy.sparse.save_npz('test_data_%i_%i.npz' % ( max_files_depr_sub, max_files_ask_sub), x_test)
    np.save('training_label_%i_%i' % ( max_files_depr_sub, max_files_ask_sub), y_train)
    np.save('test_label_%i_%i' % ( max_files_depr_sub, max_files_ask_sub), y_test)
    print("Step x: Completed.")
else:
    x_train = scipy.sparse.load_npz('training_data_%i_%i.npz' % ( max_files_depr_sub, max_files_ask_sub))
    x_test = scipy.sparse.load_npz('test_data_%i_%i.npz' % ( max_files_depr_sub, max_files_ask_sub))
    y_train = np.load('training_label_%i_%i.npy' % ( max_files_depr_sub, max_files_ask_sub))
    y_test = np.load('test_label_%i_%i.npy' % ( max_files_depr_sub, max_files_ask_sub))
#
print("Step x: Build the grid.")
grid = dict(
        alpha = [0.000005],
        max_iter = [1000],
        warm_start = [True],
        tol = [0.0000005],
        loss = ['modified_huber'],
        n_jobs = [4]
        )
print("Step x: Completed.\n")

print("Step x: Build the Logistic Regression Classifier") 
# classifier = SGDClassifier(loss='modified_huber',
#                            random_state = rand_state_nn,
#                            tol = 0.000005,
#                            n_iter_no_change = 20,
#                            verbose = 10,
#                            alpha = 0.000005,
                           
#                            penalty='l2',
#                            warm_start = True,
#                            validation_fraction = 0.1,
#                            early_stopping = True
#                            )                                   # We will use a Logistic Regression NN
classifier = SGDClassifier(random_state = rand_state_nn,
                           n_iter_no_change = 10,
                           penalty = 'l2',
                           verbose = 10)
classifier = GridSearchCV(estimator = classifier, param_grid = grid, cv = 5)
grid_result = classifier.fit( x_train, y_train)                                   # Train the NN
# 
y_predict = classifier.predict( x_test)                             # Predict the data for test
test_accuracy = classifier.score( x_test, y_test)

# Build the confusion matrix 
confMatrix = confusion_matrix(y_test, y_predict)   
tn, fp, fn, tp = confMatrix.ravel()  
# Build a classification report                       
classification_reports = classification_report( y_test, y_predict, target_names = ['Non-depressed', 'Depressed'], digits=3)

print("Best parameters: ", grid_result.best_params_)
print("Best score: ", grid_result.best_score_)
print("Test accuracy: ", test_accuracy)

# Print the confusion matrix, the classification report and other data regarding the used parameters
print(confMatrix)
print("TP - Predicted that a man is depressive and he is: %i." % ( tp))
print("TN - Predicted that a man is NOT depressive and he is NOT: %i." % ( tn))
print("FP - Predicted that a man is depressive and he is NOT: %i." % ( fp))
print("FN - Predicted that a man is not depressive and he is: %i." % ( fn))
print(classification_reports)
print("The test has been realized with the following parameters: ")
print("Random state for the training test split: %i." % ( rand_state_splitter))
print("Random state for the classifier: %i" % ( rand_state_nn))
# print("Sentences limiter: %i." % ( sentenceLimiter))
print("Data gathered from depression: (%i submission, %i comments)" % ( no_depr_sub, no_depr_comm))
print("Data gathered from AskReddit: (%i submission, %i comment)" % ( no_ask_sub, no_ask_comm))