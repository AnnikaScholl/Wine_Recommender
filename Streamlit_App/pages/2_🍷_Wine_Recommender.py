import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Plotting Settings
px.defaults.width = 800
px.defaults.height = 600
px.defaults.template = "plotly_dark"

st.set_page_config(page_title="Wine Recommender", page_icon="🍷", layout="wide")

#st.set_page_config(page_title="🍷 Wine Recommender", layout="wide")
st.title("🍷 Wine Recommender")


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


def feature_recommender(selected_features, wine_index) :
    
    st.write(
        """
        ## Your Favourite Wine
        """
    ) 

    input_wine = df_complete.filter(items=[wine_index], axis=0)[['wine_name', 'winery', 'country', 'price', 'year', 'avg_rating_wine_year', 'url']]
    st.data_editor(
        input_wine,
        column_config={
            "wine_name": "Wine Name",
            "winery": "Winery",
            "country": "Country",
            "avg_rating_wine_year": "Rating",
            "year": st.column_config.NumberColumn("Year",
            format="%d",
        ),
            "price": st.column_config.NumberColumn("Price",
            help="The price of the product in Pounds",
            format="£%d",
        ),
        "url": st.column_config.LinkColumn(
                "URL", display_text="Open distributor"
            ),
            
        },
        hide_index=True,
        use_container_width=True
    )
 
    # Get the feature vector for the wine with index e.g. "1181258_2022"
    wine_feature_vector = selected_features.loc[wine_index]

    # Get feature vectors for all other wines
    other_wines_feature_vectors = selected_features.drop(wine_index)

    # Calculate cosine similarity between the wine and all other wines
    wine_similarity_scores = cosine_similarity([wine_feature_vector], other_wines_feature_vectors)[0]

    df_wines = df_complete.drop(wine_index)

    # Create a DataFrame to hold wine indices and their similarity scores
    sim_df = pd.DataFrame({
                           'Wine Name':  df_wines['wine_name'],
                           'Winery': df_wines['winery'],
                           'Country': df_wines['country'],
                            'Similarity': wine_similarity_scores,
                            'Price': df_wines['price'],
                            'Year':  df_wines['year'],
                            'Rating':  df_wines['avg_rating_wine_year'],
                            'URL':  df_wines['url'],
                            'wine_id': df_wines['wine_id']})


    # Filtering
    # Not favourite wine form different years
    cond_1 = sim_df['wine_id'] != df_complete.loc[wine_index]['wine_id']
    # Maximum price
    cond_2 = sim_df['Price'] <= max_price
    # Minimum year
    cond_3 = sim_df['Year'] >= min_year
    # Maximum year
    cond_4 = sim_df['Year'] <= max_year
    #Minimum rating
    cond_5 = sim_df['Rating'] >= min_rating
    
    # Cechbox not same winery
    if not_same_winery:
        sim_df = sim_df[sim_df['Winery'] != df_complete.loc[wine_index]['winery']]
    
    # Checkbox not same wines from different years
    if not_same_wine:
        top_1000 = sim_df[cond_1 & cond_2 & cond_3 & cond_4 & cond_5].sort_values(by='Similarity', ascending=False).head(1000)
        top_10 = top_1000.drop_duplicates('wine_id', keep = 'first').head(x)
    else:
        top_10 = sim_df[cond_1 & cond_2 & cond_3 & cond_4 & cond_5].sort_values(by='Similarity', ascending=False).head(x)



    #Output rankings based on users selections
    st.write(
        """
        ## Your Top Wine Matches
        """
    )   

    # Display DataFrame with hyperlink column
    st.data_editor(
        top_10[['Wine Name', 'Winery', 'Country', 'Similarity', 'Price', 'Year', 'Rating', 'URL']],
        column_config={
            "Similarity": st.column_config.NumberColumn(
            help="The similarity describes how close your wine is to the other wines with 1 being the same and -1 being the opposite",
        ),
            "Year": st.column_config.NumberColumn(
            format="%d",
        ),
            "Price": st.column_config.NumberColumn(
            help="The price of the product in Pounds",
            format="£%d",
        ),
        "URL": st.column_config.LinkColumn(
                "Buy Wine", display_text="Open distributor"
            ),
            
        },
        hide_index=True,
        use_container_width=True
    )

    if st.button('Visualisation of Top Wine'):
        recommender_visualisation(top_10)
    return top_10


# Function to visualise similarity
def recommender_visualisation(df_top_10):

    # Get top match index
    top_wine_index = df_top_10.index[0]

    # 1. Taste Features
    st.header("Wine Taste Features")

    # Create the radar chart
    fig_taste = go.Figure()

    taste_values = df_complete[['body', 'taste_intensity', 'taste_tannin', 'taste_sweetness', 'taste_acidity']]
    taste_names = ['Body','Intensity','Tannin','Sweetness', 'Acidity']

    fig_taste.add_trace(go.Scatterpolar(
      r= taste_values.loc[wine_index].values,
      theta=taste_names,
      fill='toself',
      name='Favourite Wine',
    line=dict(color='lightcoral')
    ))
    fig_taste.add_trace(go.Scatterpolar(
        r= taste_values.loc[top_wine_index].values,
        theta=taste_names,
        fill='toself',
        name='Top Match',
    line=dict(color='LightSteelBlue')
    ))
    
    fig_taste.update_layout(
    font=dict(color="lightcoral", size=20),
    polar=dict(
        bgcolor='black',
          radialaxis=dict(
      visible=True,
      range=[1, 5],
      dtick=1
    )  # Set the color inside the polar chart to dark grey
    )
    )

    # Display the chart using Streamlit
    st.plotly_chart(fig_taste, use_container_width=True)

    # 2. Flavor Features
    st.header("Wine Flavour Features")

    input_wine_df = pd.DataFrame(df_complete.loc[wine_index][list(flavor_group.columns)]).reset_index()
    input_wine_df['group'] = input_wine_df['index'].str.replace('group_', '')
    input_wine_df['group'] = input_wine_df['group'].str.replace('_', ' ')

    top_wine_df = pd.DataFrame(df_complete.loc[top_wine_index][list(flavor_group.columns)]).reset_index()
    top_wine_df['group'] = top_wine_df['index'].str.replace('group_', '')
    top_wine_df['group'] = top_wine_df['group'].str.replace('_', ' ')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Favourite Wine")

        fig_fla1 = px.pie(input_wine_df, values=wine_index, names='group', color_discrete_sequence=px.colors.qualitative.Antique)
        fig_fla1.update_traces(textposition='inside', textinfo='percent+label')
        fig_fla1.update_layout(
            width=700,
            height=600,  
            showlegend=False,   
            font=dict(color="white", size=15))

        st.plotly_chart(fig_fla1, use_container_width=True)

    with col2:
        st.subheader("Top Match")

        fig_fla2 = px.pie(top_wine_df, values=top_wine_index, names='group', color_discrete_sequence=px.colors.qualitative.Antique)
        fig_fla2.update_traces(textposition='inside', textinfo='percent+label')
        fig_fla2.update_layout(
            width=700,
            height=600,  
            showlegend=False,   
            font=dict(color="white", size=15))

        st.plotly_chart(fig_fla2, use_container_width=True)

    input_wine_features = selected_features.loc[wine_index]
    other_wine_features = selected_features.loc[top_wine_index]

    # Create a DataFrame to display the comparison
    comparison_df = pd.DataFrame({'Favourite Wine': input_wine_features, 'Top Match': other_wine_features})

    # Reset index to make wine identifiers as a regular column
    comparison_df.reset_index(inplace=True)

    # Rename the index column to 'Wine'
    comparison_df.rename(columns={'index': 'Feature'}, inplace=True)

    # Melt the DataFrame to convert it to long format
    melted_df = comparison_df.melt(id_vars='Feature', var_name='Wine', value_name='Value')
    melted_df['Feature'] = melted_df['Feature'].str.replace('_', " ")
    

    # Plot the comparison using Plotly Express
    fig_scaled = px.bar(melted_df.sort_values(by='Value'), x='Value', y='Feature', color='Wine', 
                orientation='h', barmode='group', 
                color_discrete_map={"Favourite Wine": "lightcoral", "Top Match": "LightSteelBlue"})

    # Update layout to change font color and inside color of the plot
    fig_scaled.update_layout(
        xaxis_title='Values (scaled)',
        font=dict(color="lightcoral", size=20)
    )
    with st.expander("See Scaled Features"):
        st.header("All Features (scaled)")
        st.plotly_chart(fig_scaled, use_container_width=True)
    


def dataframe_with_selections(df_favorite_wine):
    df_with_selections = df_favorite_wine.drop_duplicates('wine_id', keep = 'first')

    df_with_selections.insert(0, "Select", False)

    # Get dataframe row-selections from user with st.data_editor
    edited_df = st.data_editor(
        df_with_selections[['Select', 'wine_name', 'winery', 'country', 'year', 'price','avg_rating_wine_year']],
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=True),
                       "year": st.column_config.NumberColumn(
            "Year",
            format="%d",
        ),
        'wine_name': 'Wine Name', 'winery': "Winery", 'country': "Country",'avg_rating_wine_year': "Rating",
            "price": st.column_config.NumberColumn(
            "Price",
            help="The price of the product in Pounds",
            format="£%d",
        ),},
        disabled=df.columns,
        use_container_width=True
    )

    # Filter the dataframe using the temporary column, then drop the column
    selected_rows = edited_df[edited_df.Select]

    return selected_rows.drop('Select', axis=1)


# Sidebar
add_title = st.sidebar.header("Filter your Wine ")

x = st.sidebar.slider(
    'Number of Recommendations',
    0, 30, 10
)

max_price = st.sidebar.slider(
    'Maximum Price (in £)',
    0, 300, 150, format="£%d"
)

min_rating = st.sidebar.slider(
    'Minimum Rating',
    0.0, 5.0, 0.0, step=0.1, format="%f"
)
min_year, max_year = st.sidebar.slider(
    'Select a Range of Years',
    df_complete['year'].min(), df_complete['year'].max(), (df_complete['year'].min(), df_complete['year'].max())
)

select_features_box = st.sidebar.multiselect('Choose your Features', ['Taste', 'Flavour Group'], ['Taste', 'Flavour Group'], help="To get the best recommendation, select all features.")

# Checkboxes
not_same_wine = st.sidebar.checkbox('Show Only Unique Wines', help= "Avoid getting the same wine from different years in your recommendations", value=True)

# Checkboxes
not_same_winery = st.sidebar.checkbox('Exclude Wines from the Same Winery', help= "Receive recommendations from diverse vineyards")


selected_features_list = []

if 'Taste' in select_features_box:
    selected_features_list.append(taste)
if 'Flavour Group' in select_features_box:
    selected_features_list.append(flavor_group)
if len(selected_features_list) > 0:
    selected_features = pd.concat(selected_features_list, axis=1)
else:
    st.write("Please select at least one feature.")


# Recommender Start
# Check if features are selected

if len(selected_features_list) > 0:

    # Get wine name or winery
    wine_name = st.selectbox(
        "What's your favourite wine or winery?",
        (np.unique(df_complete[['wine_name', 'winery']].values)),
            index=None,
            placeholder="Type in Your Wine or Winery...",
    )


    # Filter dataframe based on user input
    df_favorite_wine = df_complete[(df_complete['wine_name'] == wine_name) | (df_complete['winery'] == wine_name)][['wine_name', 'winery', 'country', 'year', 'price', 'avg_rating_wine_year', 'wine_id']]

    # Get if wine is not None
    if wine_name is not None:

        # Only one wine/winery with this name
        if df_favorite_wine.shape[0] == 1:
            wine_index = df_complete[(df_complete['wine_name'] == wine_name) | (df_complete['winery'] == wine_name)].index[0]
            # Checkboxes
            recommendations = feature_recommender(selected_features, wine_index)

        # More than one wine/winery with this name
        elif df_favorite_wine.shape[0] > 1:
            # Display the dataframe  
            st.write('It looks like there are a few wines with the same name or from the same winery! Please select your favourite wine.')
            selection = dataframe_with_selections(df_favorite_wine)
            if selection.shape[0] == 1:
                wine_index = selection.index[0]
                # Checkboxes
                recommendations = feature_recommender(selected_features, wine_index)
            elif selection.shape[0] > 1:
                st.write("Please only select one wine.")







