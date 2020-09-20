
import os
import re 
import json
import langdetect
from dict.contractions import contractions_dict
from dict.languages import languages

# ************************
# *** GLOBAL VARIABLES ***
# ************************
langList = languages    
template = { 
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

# *******************
# *** COUNT WORDS ***
# *******************
def countWords( text):
    return len( text.split())

# ************************
# *** COUNT CHARACTERS ***
# ************************
def countChars( text):
    counter = 0
    for char in text:
        if char != " ":
            counter += 1
    return counter

# *****************
# *** GET FILES ***
# *****************
def getFiles( directory, basename, extension):              # Define a function that will return a list of files
    pathList = []                                           # Declare an empty array
    for root, dirs, files in os.walk( directory):           # Iterate through roots, dirs and files recursively
        for file in files:                                  # For every file in files
            if os.path.basename(root) == basename:          # If the parent directory of current file is = with the parameter
                if file.endswith('.%s' % (extension)):      # If the searched file ends in the parameter
                    path = os.path.join(root, file)         # Join together the root path and file name
                    pathList.append(path)                   # Append the new path to the list
    return pathList                                         # Return the full list

# ****************************
# *** GET JSON OBJECT LIST ***
# ****************************
def packingFunction( fileList, packetSize, out, ignoreFile, ignorePatch):
    # VARIABLES
    maxCharacters = 0                                       # Maximum Characters variables
    counterLists = ignorePatch                              # Determine the starting patch
    dummyList = []                                          # a temporary list 
    dataList, statistics_packet, stats = {}, {}, {}         
    ignoreFileLocal = ignoreFile                            # Determine the starting file
    
    statistics_packet.update(template)  
    stats.update(template)
    
    textPath = os.path.join( out, "text")                   # Determine the path for the text
    logPath = os.path.join( out, "log")                     # Determine the path for the text
    
    if not os.path.exists(textPath):
        os.makedirs(textPath)
    if not os.path.exists(logPath):
        os.makedirs(logPath)
        
    # Iterate through all the files
    fileListLength = len(fileList)
    
    for file in fileList:                                   # Iterate through all the files given
        if ignoreFileLocal != 0:
            ignoreFileLocal -= 1
            continue
        with open(file, "r") as f:                          # Open the current iterated file
            print("Opening file: (%i/%i)." % (fileList.index(file), fileListLength))
            parsedData = json.loads(f.read())
            objects = parsedData['data']
            # Iterate through all the objects
            for object in objects:
                uncleanText = object['text']                # Get the 'text' from the object
                cleanText, stats = cleanData( uncleanText)  # Clear the noise from the 'text'
                object['text'] = cleanText                  # update the 'text' with the clean version
                
                # Add the received statistics to the total
                for key, value in stats.items():
                    statistics_packet[key] += stats[key]
                if object['text'] != "":
                    # Create a new lighter item    
                    dummyObject = {
                        "date": object['date'],
                        "label": object['label'],
                        "text": object['text']
                        }
                    dummyList.append(dummyObject)           # Add the cleaned item to the new list
                
                if( len(dummyList) == packetSize):          # If the packet is full, it will be outputed
                    counterLists += 1                       # increment the number of outputed lists
                    
                    # Computer words average, character average
                    statistics_packet['wordsAvg'] = statistics_packet['words'] / len(dummyList)
                    statistics_packet['charactersAvg'] = statistics_packet['characters'] / len(dummyList)
                    statistics_packet['wordLengthAvg'] = statistics_packet['characters'] / statistics_packet['words']
                    
                    dataList = {"data": dummyList}          # get the final data JSON object
                    generalList = {
                        "lastFile": fileList.index(file),
                        "lastPacket": counterLists
                        }
                    
                    dataJSON = json.dumps( dataList, sort_keys = True, ensure_ascii = True)
                    logJSON = json.dumps( statistics_packet, ensure_ascii = True)
                    generalJSON = json.dumps( generalList, ensure_ascii = True)
                    
                    pathText = os.path.join( textPath, "processed_size%i_%i.json" % ( len(dummyList), counterLists))
                    pathLog = os.path.join( logPath, "processed_size%i_%i.json" % ( len(dummyList), counterLists))
                    pathLang = os.path.join( out, "processed.json")
                    pathGeneral = os.path.join( out, "operations.json")
                    
                    fileText = open(pathText, "w")          # open a temporary stream to the text file
                    fileLog = open(pathLog, "w")            # open a temporary stream to the log file
                    fileLang = open(pathLang, "w")
                    fileGeneral = open(pathGeneral, "w")
                    
                    print( dataJSON, file = fileText)       # print the new packet of data in the file
                    print( logJSON, file = fileLog)         # print the log for the packet
                    print( langList, file = fileLang)
                    print( generalJSON, file = fileGeneral)
                    
                    for key, value in statistics_packet.items():
                        statistics_packet[key] = 0

                    print("Packet no.%i, with size %i, outputed in %s." % ( counterLists, len(dummyList), pathText))
                    print("Log no.%i outputed in %s." % ( counterLists, pathLog))
                    print("Written the general file.")
                    print("Currently the maximum characters number: %i." % (maxCharacters))
                    dummyList = []                          # reset the dummy list
                    fileText.close()                        # close the stream
                    fileLog.close()                         # close the stream
                else:
                    continue
        print("Gathered all data from: %s\n" % (file))

# ****************************
# *** CONTRACTION EXPANDER ***
# ****************************
def expandContractions( text, contractions_dict):
    contractionPattern = re.compile('({})'.format('|'.join( contractions_dict.keys())), flags=re.IGNORECASE | re.DOTALL)
    counterPos = 0
    counterNeg = 0
    def expandMatch(contraction):
        match = contraction.group(0)

        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        
        return expanded_contraction
    
    expanded_text, counterPos = contractionPattern.subn( expandMatch, text)
    expanded_text, counterNeg = re.subn("'", " ", expanded_text)
    return expanded_text, counterPos, counterNeg

# ********************
# *** DATA CLEANER ***
# ********************
def cleanData( data):
    statistics_cleaning = {}
    statistics_cleaning.update(template)
    # We need to check firstly if the Data is empty, otherwise there is no point in working with it
    if(data != ""):
        # 1. We want to do a links checking, maybe the post is an entire link, we want it removed before langdetect
        data, bufferData = re.subn(r"http\S+", " ", data)
        statistics_cleaning['links'] += bufferData

        # 2. We need to expand the contractions now, otherwise we can't delete the punctuations
        data, bufferData, bufferData2 = expandContractions( data, contractions_dict)
        statistics_cleaning['contractionsPos'] += bufferData
        statistics_cleaning['contractionsNeg'] += bufferData2
        
        # 3. Remove all the e-mail
        data, bufferData = re.subn(r"\S*@\S*", " ", data)
        statistics_cleaning['at'] += bufferData
        
        # 4. Remove punctuation             
        data, bufferData = re.subn(r"[^a-zA-Z\s]", " ", data)
        statistics_cleaning['punctuation&numbers'] += bufferData
        
        if(data != "" and not data.isspace()):
            detectedLanguage = langdetect.detect(data)                              #
            langList[detectedLanguage] += 1
            
            if(detectedLanguage == "en"):                                           #                               
                data, bufferData = re.subn(r"\s\s+", " ", data)
                statistics_cleaning['negatives'] += bufferData
                
                # 10. Transform the data to lower case
                data = data.lower()
                                
                # 11. As an extra we can count the total amount of characters
                statistics_cleaning['characters'] = countChars( data)
                
                # 12. Count the total amount of words
                statistics_cleaning['words'] += countWords( data)
                
                # Finally we return the data
                return data, statistics_cleaning
            else:
                statistics_cleaning['empty'] += 1 
                return "", statistics_cleaning
        else:
            #print("Empty data detected during the language detection phase.")
            statistics_cleaning['empty'] += 1 
            return "", statistics_cleaning
    else:
        #print("Data is empty from the beginning.")
        statistics_cleaning['empty'] += 1 
        return "", statistics_cleaning
        
# *********************
# *** MAIN FUNCTION ***
# *********************
filesList = getFiles("extracted/AskReddit/submission", "text", "json")
print("Returned %i files with .json extension" % ( len(filesList)))

packingFunction(filesList, packetSize = 1000, out="processed/AskReddit/submission", 
                ignoreFile = 1814, ignorePatch = 1189)