import streamlit as st
import plotly.express as px
import pandas as pd


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
    df = pd.read_csv('data/data_clean.csv', index_col=0)
    feature_final = pd.read_csv('data/feature_final.csv', index_col=0)
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
    flavor_subgroup = feature_scaled.iloc[:,18:-16]
    all_flavor = feature_scaled.iloc[:,5:-16]
    food = feature_scaled.iloc[:,-16:]
    taste_flavor_group = feature_scaled.iloc[:,:18]
    return taste, flavor_group, taste_flavor_group


def chart_sunburst():
    group = pd.DataFrame(df_complete.iloc[:,:13].mean(), columns=['mean']).reset_index()
    group_list = list(group['index'].str.replace('group_',''))
    subgroup_df = pd.DataFrame(df_complete.iloc[:,13:468].mean().sort_values().reset_index())
    subgroup_list = []

    for g in group_list:
        s_df = pd.DataFrame(subgroup_df['index'].str.split(g, expand=True)[1])
        s_df['weight'] = subgroup_df[0] 
        s_df.dropna(inplace=True)
        s_df['parent'] = g
        s_df.rename(columns={1: 'group'}, inplace = True)
        subgroup_list.append(s_df)

    chart = pd.concat(subgroup_list)
    chart['group'].replace(to_replace='_', value='', regex=True, inplace = True)
    chart.loc[chart['group'] == '', 'group'] = chart.loc[chart['group'] == '', 'parent']
    chart['parent'].replace(to_replace='_', value=' ', regex=True, inplace = True)
    return chart

# Load data
df, feature_final, df_complete = load_data()


# Preprocess data
feature_scaled = preprocess_data(feature_final)


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
    0.0, 4.9, 0.0, step=0.1, format="%f"
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

st.header("Average Wine Flavour Profile")
chart = chart_sunburst()

fig2 = px.sunburst(chart, path=['parent', 'group'], values='weight', color_discrete_sequence=px.colors.qualitative.Antique,
                branchvalues = 'total')
fig2.update_layout(
    width=700,
    height=700, 
    font_size=18)
st.plotly_chart(fig2, use_container_width=True)


    
