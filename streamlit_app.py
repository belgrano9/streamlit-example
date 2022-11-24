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

st.title('Project visualization')

file=st.file_uploader("Upload your file")
#stringio=StringIO(file.getvalue().decode("utf-8")
#st.write(stringio)
                              
#string_data=stringio.read()
#st.write(string_data)



