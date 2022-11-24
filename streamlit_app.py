"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

import streamlit as st
import pandas as pd
import numpy as np
import functions

st.title('Project visualization')

st.sidebar.write("Sidebar test")
st.sidebar.button("Click me")


uploaded_file=st.file_uploader("Upload your file")
if uploaded_file:
   st.write("Filename: ", uploaded_file.name)
   
filename=uploaded_file.name

st.write(functions.reading_file(filename))



