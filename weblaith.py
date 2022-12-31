import streamlit as st
from full_pipeline_project import full_pipeline

# title
st.set_page_config(page_title='laith web')

# write a title in center of the page
html_title = "<center><h1>Amazon Reviews</h1></center>"
st.write(html_title, unsafe_allow_html=True)

# ask user writing a comments
input_of_user = st.text_area(label='1', placeholder='Please write a comment...', label_visibility='hidden')

# adding a button
clicked = st.button(label='findout')

# Result of the model
result = full_pipeline(input_of_user)
print(result)
# show dataframe
import pandas as pd
st.dataframe(result)  # pd.DataFrame({'sad':[1, 2, 3, 4], 'kill':[10, 20, 30, 40]}))
# if clicked:
#     st.table(result)

