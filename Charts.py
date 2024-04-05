import streamlit as st
import pandas as pd
import numpy as np

st.header('Dan did a thing. And Meg was amazed. Its just a Line chart but its coded from scratch')

chart_data = pd.DataFrame(
     np.random.randn(20, 4),
     columns=['a', 'b', 'c','d'])

st.line_chart(chart_data)
