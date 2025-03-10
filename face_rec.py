import pandas as pd
import numpy as np
import cv2
import redis

# Insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

## time
import time
from datetime import datetime

# Connect to Redis Client
hostname = 'redis-19584.c240.us-east-1-3.ec2.redns.redis-cloud.com'
portnumber = 19584
password = 'wdrBP3DOyw1YfBIS7dsUSUvZxiGJiNSI'
# redis-19584.c240.us-east-1-3.ec2.redns.redis-cloud.com:19584
#  wdrBP3DOyw1YfBIS7dsUSUvZxiGJiNSI

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

############## Retrive Data from Database ################

def retrive_data(name):
    retrive_dict= r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))### if face is detect unknow see the refernece in 4-Prediction file in jupyter notebook
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','facial_feature'] ### this line is used to give the index which is name_role and facial_feature
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
    ### this above line just used to make a separate name and the role
    return retrive_df[['Name','Role','facial_feature']]


# Configure Face Analysis
faceapp = FaceAnalysis(name='buffalo_l',root="E:/MAJOR PROJECT/buffalo_l",providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

### Warning: we don't have to set the det_thresh < 0.3


############################################## ML SEARCH ALGORITHM ##############################################


def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['Name','Role'],thresh=0.62):  #### In the video it use 0.5 because that time 0.5 
    #### is max but now 
    ### we have the data which have 0.6 we can check in the above graphs ,that is we see clearly that the line is going to till 0.61 
    #####"""""""THAT IS WHY WE ARE USING THE THRESHOLD 0.62""""""####################
    #####  COSINE SIMILARITY BASE SEARCH ALGORTIHM  ####################

#### Step-1: Take the Data Frame (Collection of DATA)
    dataframe = dataframe.copy()

##### Step-2: Index Face Embeding from the Data Frame and convert into Array
    X_list = dataframe[feature_column].tolist()
    fixed_X_list = []
    for embedding in X_list:
        if len(embedding) > 512:
            fixed_embedding = embedding[:512]  # Truncate to length 512
        elif len(embedding) < 512:
            fixed_embedding = np.pad(embedding, (0, 512 - len(embedding)), 'constant')  # Pad with zeros to length 512
        else:
            fixed_embedding = embedding
        fixed_X_list.append(fixed_embedding)
    x = np.asarray(fixed_X_list)

#### Step-3: Calculating the Cosine Similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1)) ##### ""-1""" is explain in the above things so you/we can check it out there
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr


#### Step-4: Filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:

        
        #### Step-5: Get the person name /Person
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]

    else:
        person_name = 'Unknown Person'
        person_role = 'Unknown Role'


    return person_name, person_role    

####################### Real-time-prediction #######################
### We need to save "Logs" for every 1 minute.
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def saveLogs_redis(self):
        ### Step-1: Create a logs dataframe
        dataframe = pd.DataFrame(self.logs)

        ## Step-2: Drop the Duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True)

        # Step-3: Save the data/ Push the data into redis database  (list)
        ## We will encode the data
        name_list = dataframe['name'].to_list()
        role_list = dataframe['role'].to_list()
        ctime_list = dataframe['current_time'].to_list()

        # We will take the loop
        encoded_data = []
        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != 'Unknown Person':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) >0: #if len > 0 then the data will go to redis database.
            r.lpush('attendance:logs',*encoded_data)## this line tells we push the data in redis database.

        self.reset_dict()    


########################################## TEST IMAGE TO TAKE IN THE ML SEARCH ALGORITHM ##################################

    def face_prediction(self,test_image, dataframe,feature_column,name_role=['Name','Role'],thresh=0.62):
        ## Step-0: find the time
        current_time = str(datetime.now())
        ### Step-1: Take the test image and apply to the insight face
        
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        
        ### Step-2: Use for loop and extract each embedding and pass to ml_search_algorithm
        
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
        
            if person_name == 'Unknown Person':
                color =(0,0,255) #### BGR(Blue,Green,Red)
            else:
                color =(0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2) ### 0.7 is a FONT SIZE OF THE WORDS/CHARACTERS
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)

            ###### Save information in "Logs" dict.
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


### Registration Form ###
class RegistrationForm:
    def _init_(self):
        self.sample = 0
    def reset(self):
        self.sample = 0

    def get_embedding(self,frame):
        # Get results from Insight Face Model
        results = faceapp.get(frame,max_num=1) ### max_num =1 means we are going to select the one particular face
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)

                # FACIAL Features
            embeddings = res['embedding']

        return frame, embeddings    