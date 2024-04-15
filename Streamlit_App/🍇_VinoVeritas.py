import streamlit as st

st.set_page_config(
    page_title="VinoVeritas",
    page_icon="🍇",
    layout="wide"
)

st.write("# Welcome to VinoVeritas! ")

st.markdown(
    """
    ### 🍷 Problem Statement

    Ever find yourself lost in the wine aisle, surrounded by countless options but unsure which one will suit your taste buds? 
    With so many wines lining the shelves, it's easy to feel overwhelmed. 🛒

    While traditional review platforms may offer ratings and recommendations based on wineries, they often overlook the nuances of individual preferences. 
    However, instead of focusing solely on wine scores, the real challenge lies in navigating the vast array of flavours and tastes available. 

    ### 💡 Solution

    That's where VinoVeritas comes in! 🚀 
    Its solution isn't just about scores; it's about matching you with wines that suit your taste buds in all the right ways. 
    Say goodbye to guesswork and hello to a personalised, content-based recommendations system!

    Using cosine similarity, VinoVeritas delves into the intricate flavours and taste characteristics of your favorite wine. 
    From the richness of the body to the subtle hints of oak and red fruit, it analyses it all to find wines that match your preferences perfectly.

    
    So the next time you find yourself lost in the wine aisle, let VinoVeritas be your guide. With it by your side, discovering your new favorite bottle has never been easier. 
    
    Cheers to endless adventures in wine tasting! 🍷✨
"""
)
st.subheader("Project Overview")
st.image('data/wine.png')
