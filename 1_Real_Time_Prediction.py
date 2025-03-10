##################### just try to change the float value to32 & 64 for my face & old dataset face respectively in ""face_rec.py""" ####
import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import logging
import time

# Set up logging
logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title='Prediction')
st.subheader("Real-Time Attendance System")

## Step-1: Retrieve the data from Redis Database ###
with st.spinner('Retrieving Data from Redis DB....'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success("Data Successfully retrieved from Redis")

###### Time #############
waitTime = 30 # time in sec
setTime = time.time()
realtimepred = face_rec.RealTimePred() #### Real time prediction class


## Step-2: Real-Time Prediction ###
# Streamlit webrtc

# Call-back function
def video_frame_callback(frame):
    try:
        global setTime
        img = frame.to_ndarray(format="bgr24")
        logging.debug("Frame received")

        pred_img = realtimepred.face_prediction(img, redis_face_db, 'facial_feature', ['Name', 'Role'], thresh=0.62)

        timenow = time.time()
        difftime = timenow - setTime
        if difftime >= waitTime:
            realtimepred.saveLogs_redis()
            setTime = time.time() #reset time
            print('Save Data to redis Database')
        logging.debug("Frame processed")

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    except Exception as e:
        logging.error(f"Error processing frame: {e}")
        return frame

# Start the video stream without STUN server configuration
webrtc_ctx = webrtc_streamer(
    key="RealTimePrediction",
    video_frame_callback=video_frame_callback,
)

if webrtc_ctx.state.playing:
    st.write("Video stream is live")
else:
    st.write("Video stream not started")
