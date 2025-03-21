import streamlit as st 
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')


##  Initialize Registration Form ### 
registration_form = face_rec.RegistrationForm()


## Step-1 : Collect Person name and Role
# form
person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select Your Role',options=('Student','Teacher'))


## Step-2 : Collect facial Embedding of that person
def video_callback_fun(frame):
    img = frame.to_ndarray(format='bgr24')
    reg_img, embedding = registration_form.get_embedding(img)

# Two step process 
# First step -----> save data into local computer with the ".txt" format.
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')
webrtc_streamer(key='registration',video_frame_callback=video_callback_fun)

## Step-3 : Save the data in redis database.

if st.button('Submit'):
    st.write(f'Person Name = ',person_name)
    st.write(f'Role = ',role)