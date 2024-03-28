import streamlit as st
st.header('This is a test to show a button (st.button) in action')

st.button("Reset", type="primary")
if st.button('Say hello to Streamlit app :smile:'):
    st.write('Wahey, it works')

else:
    st.write('Goodbye sucker')
