import streamlit as st
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

px.defaults.width = 800
px.defaults.height = 600
px.defaults.template = "plotly_dark"


st.set_page_config(page_title="Wine Insights", page_icon="🔍", layout="wide")


st.title("🔍 Wine Insights")

@st.cache_data
def load_data():
    df = pd.read_csv('Streamlit_App/data_app/data_clean.csv', index_col=0)
    feature_final = pd.read_csv('Streamlit_App/data_app/feature_select.csv', index_col=0)
    df_complete = feature_final.iloc[:,5:].join(df, how="inner")
    return df, feature_final, df_complete


@st.cache_data
def preprocess_data(feature_final):
    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_final)
    feature_scaled = pd.DataFrame(data=X, columns=feature_final.columns, index=feature_final.index)
    return feature_scaled

@st.cache_data
def features(feature_scaled):
    taste = feature_scaled.iloc[:,:5]
    flavor_group = feature_scaled.iloc[:,5:18]
    return taste, flavor_group


# Load data
df, feature_final, df_complete = load_data()


# Preprocess data
feature_scaled = preprocess_data(feature_final)

# Get features
taste, flavor_group = features(feature_scaled)

# Sidebar Filter
country = st.sidebar.selectbox(
    'Choose a country:',
    (np.sort(df_complete['country'].unique())),
        index=None,
        placeholder="Select country...",
)

if country is not None:
    df_complete = df_complete[df_complete['country'] == country]


min_rating = st.sidebar.slider(
    'Minimum Rating',
    0.0, round(df_complete['avg_rating_wine_year'].max(),1), 0.0, step=0.1, format="%f"
)

df_complete = df_complete[df_complete['avg_rating_wine_year'] >= min_rating]

min_year, max_year = st.sidebar.slider(
    'Select a Range of Years',
    df_complete['year'].min(), df_complete['year'].max(), (df_complete['year'].min(), df_complete['year'].max())
)

df_complete = df_complete[(df_complete['year'] >= min_year) & (df_complete['year'] <= max_year)]



st.header("Average Wine Taste Profile")
# Create taste_chart DataFrame
taste_chart = pd.DataFrame(dict(
    r=[df_complete['body'].mean(), df_complete['taste_intensity'].mean(), df_complete['taste_tannin'].mean(), df_complete['taste_sweetness'].mean(), df_complete['taste_acidity'].mean()],
    theta=['Body','Intensity','Tannin',
        'Sweetness', 'Acidity']))

# Create the polar chart
fig = px.line_polar(taste_chart, r='r', theta='theta', line_close=True)

# Update traces with lighter red font color and line color
fig.update_traces(fill='toself', line=dict(color='lightcoral'))

# Update layout to change font color and inside color of the polar chart
fig.update_layout(width=500,
    height=500,
    font=dict(color="lightcoral", size=20),
    polar=dict(
        bgcolor='black',
          radialaxis=dict(
      visible=True,
      range=[1, 5],
      dtick=1
    )  
    )
)
# Display the chart using Streamlit
st.plotly_chart(fig, use_container_width=True)


st.markdown("***")


st.header("Average Wine Flavour Profile")

input_wine_df = pd.DataFrame(df_complete[list(flavor_group.columns)]).mean().reset_index()
input_wine_df['group'] = input_wine_df['index'].str.replace('group_', '')
input_wine_df['group'] = input_wine_df['group'].str.replace('_', ' ')

fig_fla1 = px.pie(input_wine_df, values=0, names='group', color_discrete_sequence=px.colors.qualitative.Antique)
fig_fla1.update_traces(textposition='inside', textinfo='percent+label')
fig_fla1.update_layout(
    width=700,
    height=600,  
    showlegend=False,   
    font=dict(color="white", size=15))

st.plotly_chart(fig_fla1, use_container_width=True)
    
