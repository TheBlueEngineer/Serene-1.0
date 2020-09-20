# **************************
# *** IMPORTED LIBRARIES ***
# **************************
import requests                    
import json
import time
import datetime 
import os
import math

PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"

# ********************
# *** EXTRACT DATA ***
# ********************
def query(**kwargs):
    # Default paramaters for API query which are necessary no matter 
    params = {
        "sort_type":"created_utc",                  # Sort type: By the time of the submission in UTC zone
        "sort":"asc",                               # Sort: Ascended Order
        "size":1000                                 # Size: 1000 is the maximum number of query entries
        }
    
    # Add additional paramters based on function arguments
    for key, value in kwargs.items():               # Key => Value mapping, according to KWARGS arguments
        params[key] = value                         # The array of parameters get new values for each new key
    
    # Set the type variable based on function input, type can be "comment" or "submission", default is "comment"
    if 'type' in kwargs and kwargs['type'].lower() == "submission":
        type = "submission"
    else:
        type = "comment"
    
    # Perform an API request to Pushshift.io
    Request = requests.get(PUSHSHIFT_REDDIT_URL + "/" + type + "/search/", params=params, timeout=30)
    
    # Print the request status code
    print("Request status code: %i." % (Request.status_code))
    # Check the status code, if successful, process the data
    if Request.status_code == 200:                          # 200 is the status code OK, which means succesful operation
        response = json.loads(Request.text)                 # Load the text from the request in JSON format
        data = response['data']                             # Get the Data from the response
        sorted_data_by_id = sorted( data, key = lambda x: int( x['id'], 36))
        return sorted_data_by_id                            # Return the data sorted by ID

# ********************
# *** EXTRACT DATA ***
# ********************
def extract_reddit_data(**kwargs):
    # Speficify the start timestamp
    max_created_utc = kwargs['startDate']                   # Specify the "after" date 
    if 'endDate' in kwargs:                                 # If we have a "before" date, we take it in consideration
        min_created_utc = kwargs['endDate']                 # 
    else:
        min_created_utc = math.floor( time.time())          # Create epoch timestamp for min date
    max_id = 0                                              # Declare the ID
    
    statistics_template = {
        "removed": 0,
        "deleted": 0,
        "empty": 0,
        "error": 0,
        "sticky": 0,
        "unusable": 0,
        "usable": 0
    }
    
    statistics_messages = {
        "removed": "Data removed by admin: ",
        "deleted": "Data deleted by poster: ",
        "empty": "Empty body of text: ",
        "error": "Ignored due to errors: ",
        "sticky": "Stickied/pinned posts ignored: ",
        "unusable": "Unusable data: ",
        "usable": "Usable data: ",
    }
    # total counters for the data
    statistics_local, statistics_total = {}, {}
    statistics_local.update( statistics_template)
    statistics_total.update( statistics_template)
    
    # To make data easier to be accessed or modified, we want to create two folders, "comment" and "submission"
    directory = os.path.join( "extracted", kwargs['subreddit'], kwargs['type'])
    if not os.path.exists( directory):                                  # IF the base dir doesn't exist THEN
        os.makedirs( directory)                                         # 
        
    # ************************************************
    # *** THE MAIN WHILE, WHERE EVERYTHING HAPPENS ***
    # ************************************************
    while 1:
        nothing_processed = True                                        # In case of no process we want to quit the while
        # Call the recursive fuction to get the batch of 1000 JSON objects each time
        objects = query(**kwargs,                                # specify that it will receive KWARGS
                        after = max_created_utc,                 # We get the objects after this date
                        before = min_created_utc)                # We get the objects before this date
        # Print the current date from which the data is gathered and the maximum date
        print("Trying to get data after %i and before %i." % (max_created_utc, # The date after we search
                                                              min_created_utc))# The date before we search
        # Transform the epoch date in a normal date format
        FormatedDate = datetime.datetime.utcfromtimestamp(max_created_utc)
        FormatedDateStr = ( "%i.%i.%i_%i.%i" % (FormatedDate.day,       # Day
                                                FormatedDate.month,     # Month
                                                FormatedDate.year,      # Year
                                                FormatedDate.hour,      # Hour
                                                FormatedDate.minute))   # Minute
        # Print an information to know that we started a new batch of data
        print("Extracting the next %i %s(s) from %s subreddit, from date: %s (%s)" % (1000, kwargs['type'], kwargs['subreddit'], FormatedDate, max_created_utc))
        
        # Now its ideal to create a folder for each year, for each one of the two types 
        directoryDate = os.path.join( directory, str(FormatedDate.year))# Entire path where the date directory is written
        directoryText = os.path.join( directoryDate, "text")            # Entire path for text in directory date
        directoryLog = os.path.join( directoryDate, "log")              # Entire path for log in directory date
        
        if not os.path.exists( directoryDate):                          # IF the date directory exists
            os.mkdir( directoryDate)                                    # write the directory date
            os.mkdir( directoryText)                                    # write the text directory in directory date
            os.mkdir( directoryLog)                                     # write the log directory in directory date
            
        # fileName = [Subreddit]/ [comment / submission]/ [Date]/ [Text/Log]/ [Date].json
        fileName = "%s/%s.json" % ( directoryText, FormatedDateStr)     # Determine the name of the file where we want to write
        
        fileWrite = open( fileName, "w")                                # Determine the file where we want to write 
        
        generalLogFile = open( "generalLog.txt", "w")
        
        # We have to take into consideration that the list might be of type "None"
        if objects is None:
            print("A batch of 1000 objects at time %i could not be retrieved because of a \"NoneType\" error." % ( max_created_utc))
            print("Trying again.")
            print("ERROR NONE-TYPE: on current UTC %i." % (max_created_utc), file = generalLogFile)
        else:
            # Loop the returned data, ordered by date
            listData = []                                       # Start a new list of Data of the next 1000 batch
            for object in objects:
                checked = False                                 # Used to determine if the object will be written in file
                id = int( object['id'], 36)                     # get the ID of the object
                
                if id > max_id:                                 # Since we ordered the data by ID, this is mandatory
                    nothing_processed = False                   # Since we found useful data, we don't need to quit while     
                    created_utc = object['created_utc']         # Return the date of creation for the object
                    max_id = id                                 # get the ID of the object
                    
                    if created_utc > max_created_utc:           # If the current creation date is bigger than max
                        max_created_utc = created_utc           # replace the current maximum creation date
                    
                    # If the data is of type "submission" 
                    if kwargs['type'] == "submission":
                        # Check if the submission is pinned, we don't want them
                        if 'pinned' in object.keys() and object['pinned'] == True:
                                statistics_local['sticky'] += 1
                                statistics_local['unusable'] += 1
                        elif 'stickied' in object.keys() and object['stickied'] == True:
                                statistics_local['sticky'] += 1
                                statistics_local['unusable'] += 1
                        # Check if there is an actual text in the submission
                        elif 'selftext' not in object.keys():
                            statistics_local['error'] += 1
                            statistics_local['unusable'] += 1
                        # check if the submissions has been removed
                        elif object['selftext'] in {"[removed]"}:
                            statistics_local['removed'] += 1
                            statistics_local['unusable'] += 1
                        # check if the submissions has been deleted
                        elif object['selftext'] in {"[deleted]"}:
                            statistics_local['deleted'] += 1
                            statistics_local['unusable'] += 1
                        else:
                            if object['selftext'] in {""}:
                                statistics_local['empty'] += 1
                            newData = {
                                "id": id,                           # the ID of the object
                                "date": object['created_utc'],
                                "title": object['title'],
                                "text": object['selftext'],          # Get the text of the submission/comment
                                "label": kwargs['label']
                            }
                            checked = True
                            statistics_local['usable'] += 1                       # Increment the local usable posts counter 
                            
                    # If the data is of type "comment"        
                    elif kwargs['type'] == "comment":
                        # Check if there is an actual text in the submission
                        if 'stickied' in object.keys() and object['stickied'] == True:
                            statistics_local['sticky'] += 1
                            statistics_local['unusable'] += 1
                        elif 'body' not in object.keys():
                            statistics_local['error'] += 1
                            statistics_local['unusable'] += 1
                        # check if the submissions has been removed
                        elif object['body'] in {"[removed]"}:
                            statistics_local['removed'] += 1
                            statistics_local['unusable'] += 1
                        # check if the submissions has been deleted
                        elif object['body'] in {"[deleted]"}:
                            statistics_local['deleted'] += 1
                            statistics_local['unusable'] += 1
                        # check if the submissions is empty
                        elif object['body'] in {""}:
                            statistics_local['empty'] += 1
                            statistics_local['unusable'] += 1
                        else:
                            newData = {                         # 
                                "id": id,                       # the ID of the object
                                "date": object['created_utc'],  # The date in which the data was posted
                                "text": object['body'],         # Get the text of the submission/comment
                                "label": kwargs['label']        # 1 = depression, 0 = non-depressed
                            }     
                            checked = True
                            statistics_local['usable'] += 1                       # Increment the local usable posts counter 
                    # If checked is TRUE we write the object
                    if(checked):
                        listData.append(newData)

            # *************************************************
            # *** WRITE THE ENTIRE DATA JSON OBJECT IN FILE ***
            # *************************************************            
            data = {"data": listData}
            newJSONObject = json.dumps( data, sort_keys = True, ensure_ascii = True)
            print( newJSONObject, file = fileWrite)
            
            # ***********************************************
            # *** UPDATE THE COUNTER AND PRINT IN CONSOLE ***
            # ***********************************************
            for key, value in statistics_local.items():
                # Update the total statistics values 
                statistics_total[key] += statistics_local[key]
                print(statistics_messages[key],         
                      statistics_local[key],
                      "(Total:",statistics_total[key],")"
                      )
            
            # ***************************************************
            # *** WRITE THE LOG FOR THE CURRENT BATCH OF 1000 ***
            # ***************************************************
            # Establish a file connection
            localLogName = "%s/%s.json" % ( directoryLog, FormatedDateStr)
            localLogFile = open ( localLogName, "w")

            # Create the JSON object
            JSONObjectLog = json.dumps( statistics_local, ensure_ascii = True)
            # Print the local statistics in log file
            print( JSONObjectLog, file = localLogFile)
            print( "Statistics written in log file %s.json\n" % (FormatedDateStr))
            # Reset the current total counters for the current 1000 objects
            for key, value in statistics_local.items():
                statistics_local[key] = 0
            
            # Exit if nothing happened
            if nothing_processed:
                max_created_utc -= 1
                print("Nothing left to process.")    
                break
            
            # Sleep a little before the next recursive function call
            time.sleep(1)
    # We did it, boys ! 

# ************
# *** MAIN ***
# ************
# Start program by calling function with:
# 1) Subreddit specified
# 2) The type of data required (comment or submission) 
# 1230768000  # 01/01/2009 @ 12:00am (UTC) - depression
# 1580515200  # 01/02/2020 @ 12:00am (UTC) - depression ending
# 1201219200  # 25/01/2008 @ 12:00am (UTC) - happy
# 1201219200  # 25/01/2008 @ 12:00am (UTC) - AskReddit
extract_reddit_data(subreddit="AskReddit",type="comment",startDate = 1484956800, 
                    endDate=1580515200, label = 0)