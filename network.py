import pandas as pd
import numpy as np
import os
import json
import string
import nltk
import random as python_random
from stopwords import stopwords_partial

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

# ************************
# *** GLOBAL VARIABLES ***
# ************************
alphabet = "abcdefghijklmnopqrstuvwxyz"
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)

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

def stringVectorizer( text, alphabet = string.ascii_lowercase):
    newList = []
    itteration = 0
    for sentence in text:
        itteration += 1
        print("String Vectorizer: (%i/%i)" % ( itteration, len(text)))
        vector = [[0 if char != letter else 1 for char in alphabet] for letter in text]
        newList.append(vector)
    return newList

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

def LogisticRegressionModel(x_train, y_train, x_test, y_test, maxIter):
    classifier = LogisticRegression( max_iter = maxIterations, verbose = 1)    # We will use a Logistic Regression NN
    classifier.fit( x_train, y_train)                             # Train the NN
    y_predict = classifier.predict(x_test)                        # Predict the data for test

    # Build the confusion matrix 
    confMatrix = confusion_matrix(y_test, y_predict)     
    # Build a classification report                       
    classification_reports = classification_report( y_test, y_predict, target_names = ['Non-depressed', 'Depressed'], digits=3)
    
    # Print the confusion matrix, the classification report and other data regarding the used parameters
    print(confMatrix)
    print(classification_reports)
    print("The test has been realized with the following parameters: ")
    print("Random state for the training test split: %i." % ( rand_state_splitter))
    print("Random state for the classifier: %i" % ( rand_state_nn))
    print("Sentences limiter: %i." % ( sentenceLimiter))
    print("Data gathered from depression: (%i submission, %i comments)" % ( no_depr_sub, no_depr_comm))
    print("Data gathered from AskReddit: (%i submission, %i comment)" % ( no_ask_sub, no_ask_comm))

# ***********************
# *** KERAS MODEL CNN ***
# ***********************
def KerasModel( filter_size, kernel_size, vocab_size, embedding_size, maxLengthSentences, hidden_dim):
    print("Building the CNN")
    model = Sequential([
        Embedding(input_dim = vocab_size,               
                  output_dim = embedding_size,
                  input_length = maxLengthSentences),
        Dropout(0.5),
        Conv1D(filter_size,
               kernel_size,
               padding='valid',
               activation='relu'
                ),
        MaxPooling1D(),
        Flatten(),
    	Dense( hidden_dim, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
        ])    
    print("CNN built.")
    print("Compiling the CNN.")
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy']
                )
    print("Step 8: Completed.\n")
    model.summary()    
    
    return model
# ************
# *** MAIN ***
# ************
# Maximum current possible files: 750/x depression, 1310/x AskReddit
max_files_depr_sub, max_files_depr_comm = 100,0                          # Max files you want to get from Depression category
max_files_ask_sub, max_files_ask_comm = 100,0                            # Max files you want t0 get from AskReddit category

paths = ["processed/depression/submission",
         "processed/depression/comment", 
         "processed/AskReddit/submission", 
         "processed/AskReddit/comment"]

testSize = 0.2
maxIterations = 300
sentenceLimiter = 0
rand_state_splitter = 1000
rand_state_nn = 1000
maxLengthSentences = 1000
embedding_size = 15
kernel_size = 5
filter_size = 64
hidden_dim = 128
epoch_size = 20
batch_size = 64
maxVocabulary = 5000

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

# **********************
# *** SPLIT THE DATA ***
# **********************
text = pd_final['text'].values
label = pd_final['label'].values

print("Step 3.x: Cleaning the data of stopwords to limit the data.")

tokenizer = Tokenizer(num_words = maxVocabulary)
tokenizer.num_words = 5000
tokenizer.fit_on_texts(text)

print("Step 4: Splitting the data for training and testing")
sentences_train, sentences_test, y_train, y_test = train_test_split(text, label, test_size = testSize, shuffle = True, random_state = rand_state_splitter)
print("Step 4: Completed.\n")

print("Step: 5: Text to sequence process for the training data.")
x_train = tokenizer.texts_to_sequences(sentences_train)
print("Step: 5: Text to sequence process for the testing data.")
x_test = tokenizer.texts_to_sequences(sentences_test)
print("Step 5: Completed.\n")

print("Step 6: Padding/Cutting the training data to the length of %i" % (maxLengthSentences))
x_train = pad_sequences( x_train, padding = 'post', maxlen = maxLengthSentences)
print("Step 6: Padding/Cutting the testing data to the length of %i" % (maxLengthSentences))
x_test = pad_sequences( x_test, padding = 'post', maxlen = maxLengthSentences)
print("Step 6: Completed.\n")

vocab_size = maxVocabulary + 1
print("Vocabulary size: %i." % ( vocab_size))

param_grid = dict(filter_size = [32],
                  kernel_size = [5],
                  vocab_size = [maxVocabulary],
                  embedding_size = [5],
                  epochs = [10],
                  hidden_dim = [64],
                  batch_size = [32],
                  maxLengthSentences = [ maxLengthSentences])

model = KerasClassifier(build_fn = KerasModel, verbose= True)
    
grid = GridSearchCV(estimator = model, 
                    param_grid = param_grid,
                    cv = 2, 
                    verbose = 1,  
                    return_train_score = True)
grid_result = grid.fit(x_train, y_train)

# Evaluate testing set
test_accuracy = grid.score(x_test, y_test)
print("Grid result best score:")
print(grid_result.best_score_)
print("Grid result best parameters:")
print(grid_result.best_params_)
print("Test accuracy done on x_test and y_test: ")
print(test_accuracy)

y_predict = grid.best_estimator_.predict(x_test)
confusionMatrix = confusion_matrix( y_test, y_predict)
print(confusionMatrix)
classification_reports = classification_report( y_test, y_predict, target_names = ['Non-depressed', 'Depressed'], digits=3)
print(classification_reports)