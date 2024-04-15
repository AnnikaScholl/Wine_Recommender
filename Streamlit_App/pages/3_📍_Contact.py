import streamlit as st
import pandas as pd


st.set_page_config(page_title="Contact", page_icon="📍", layout="wide")

st.markdown(
    """
    ### About Me

    I'm Annika, a Data Scientist with an educational background in business engineering. 
    Through my previous internships, I have gained practical experience in analysing complex data, collecting data, and automating processes in a fast-paced environment, 
    which has led me to the world of data. My top skills are Python, SQL, and data visualisation tools such as Tableau and Looker.
    I bring structure, dedication, and a can-do attitude to every project. Let's make magic happen together! ✨
    
    ### Contact Information

    📩 Email: scholl-annika@web.de
"""
)
url = "https://www.linkedin.com/in/annika-scholl/"
st.markdown("🔗 LinkedIn: [Annika](%s)" % url)

st.subheader("Want to learn more?")
#st.markdown("Want to learn more?")
with open("data_app/Annika_Scholl_CV.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()
st.download_button(label="Download my CV here!📄",
                   data = PDFbyte,
                    file_name="data_app/Annika_Scholl_CV.pdf",
                    mime='application/octet-stream')
