import streamlit as st
st.header('st.button')

st.button("Reset", type="primary")
if st.button('Say hello to Streamlit app :smile:'):
    st.write('Wahey, it works')

else:
    st.write('Goodbye sucker')