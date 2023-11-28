import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

import ydata_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()
sns.set(style="whitegrid")
#%matplotlib inline
from scipy.stats import zscore
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import requests
from io import StringIO

#setting up for customized printing
from IPython.display import Markdown, display
from IPython.display import HTML

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h2 style='text-align: center;'>Advanced Recommendation Engine</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Enhancing E-Commerce Recommendations with Database Optimization and Collaborative Filtering</h3>", unsafe_allow_html=True)
st.markdown("<p>Team Members: <i>Aniruddha Chakravarty, Sai Mounika Peteti, Sumukh Naveen Aradhya</i></p><br><br>", unsafe_allow_html=True)
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))

#function to display dataframes side by side
from IPython.display import display_html
def display_side_by_side(args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline;margin-left:50px !important;margin-right: 40px !important"'),raw=True)

def distplot(figRows,figCols,xSize, ySize, data, features, colors, kde=True, bins=None):
    f, axes = plt.subplots(figRows, figCols, figsize=(xSize, ySize))

    features = np.array(features).reshape(figRows, figCols)
    colors = np.array(colors).reshape(figRows, figCols)

    for row in range(figRows):
        for col in range(figCols):
            if (figRows == 1 and figCols == 1) :
                axesplt = axes
            elif (figRows == 1 and figCols > 1) :
                axesplt = axes[col]
            elif (figRows > 1 and figCols == 1) :
                axesplt = axes[row]
            else:
                axesplt = axes[row][col]
            plot = sns.distplot(data[features[row][col]], color=colors[row][col], bins=bins, ax=axesplt, kde=kde, hist_kws={"edgecolor":"k"})
            plot.set_xlabel(features[row][col],fontsize=20)

def scatterplot(rowFeature, colFeature, data):
    f, axes = plt.subplots(1, 1, figsize=(10, 8))

    plot=sns.scatterplot(x=rowFeature, y=colFeature, data=data, ax=axes)
    plot.set_xlabel(rowFeature,fontsize=20)
    plot.set_ylabel(colFeature,fontsize=20)

import numpy as np
electronics = '/Users/sumukharadhya/Downloads/ratings_Electronics.csv'

import pandas as pd
import io
#filename = list(electronics.keys())[0]
electronics = pd.read_csv(electronics, names=['userId', 'productId', 'Rating','timestamp'], header=None)

# Now the 'electronics' DataFrame should be the same as if you had read it from the local file 'ratings_Electronics.csv'
print(electronics.head())

electronics.drop('timestamp', axis=1, inplace=True)

electronics_groupby_users_Ratings = electronics.groupby('userId')['Rating']
electronics_groupby_users_Ratings = pd.DataFrame(electronics_groupby_users_Ratings.count())
user_list_min50_ratings = electronics_groupby_users_Ratings[electronics_groupby_users_Ratings['Rating'] >= 50].index
electronics =  electronics[electronics['userId'].isin(user_list_min50_ratings)]

train_data, test_data = train_test_split(electronics, test_size =.30, random_state=10)
printmd('**Training and Testing Set Distribution**', color='brown')

print(f'Training set has {train_data.shape[0]} rows and {train_data.shape[1]} columns')
print(f'Testing set has {test_data.shape[0]} rows and {test_data.shape[1]} columns')



def notebook_cell_1():
    st.write(electronics.head())

def notebook_cell_2():
    st.write('The total number of rows :', electronics.shape[0])
    st.write('The total number of columns :', electronics.shape[1])

def notebook_cell_3():
    st.write(display(electronics[['Rating']].describe().transpose()))

def notebook_cell_4():
    pal = sns.color_palette(palette='Set1', n_colors=16)
    import numpy as np
    st.pyplot(distplot(1, 1, 10, 7, data=electronics, features=['Rating'], colors=['blue']))
    st.pyplot(distplot(1, 1, 10, 7, data=electronics, features=['Rating'], colors=['red'], kde=False))

def notebook_cell_5():
    electronics_groupby_products_Ratings = electronics.groupby('productId')['Rating']
    electronics_groupby_products_Ratings.count().clip(upper=30).unique()
    ratings_products = pd.DataFrame(electronics_groupby_products_Ratings.count().clip(upper=30))
    ratings_products.rename(columns={"Rating": "Rating_Count"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings_products, features=['Rating_Count'], colors=['green'], kde=False))

def notebook_cell_6():
    electronics_groupby_users_Ratings = electronics.groupby('userId')['Rating']
    electronics_groupby_users_Ratings.count().clip(lower=50).unique()
    rating_users = pd.DataFrame(electronics_groupby_users_Ratings.count().clip(lower=50, upper=300))
    rating_users.rename(columns={"Rating": "Rating_Count"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=rating_users, features=['Rating_Count'], colors=['orange'], kde=False, bins=50))

def notebook_cell_7():
    ratings = pd.DataFrame(electronics.groupby('productId')['Rating'].mean())
    ratings.rename(columns={"Rating": "Rating_Mean"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50))


def notebook_cell_8():
    ratings = pd.DataFrame(electronics.groupby('productId')['Rating'].mean())
    ratings['Rating_Count'] = electronics.groupby('productId')['Rating'].count()
    st.pyplot(scatterplot('Rating_Mean', 'Rating_Count', data=ratings))

def notebook_cell_9():
    ratings = pd.DataFrame(electronics.groupby('userId')['Rating'].mean())
    ratings.rename(columns={"Rating": "Rating_Mean"}, inplace=True)
    st.pyplot(distplot(1, 1, 10, 7, data=ratings, features=['Rating_Mean'], colors=['brown'], kde=False, bins=50))

def notebook_cell_10():
    ratings = pd.DataFrame(electronics.groupby('productId')['Rating'].mean())
    ratings['Rating_Count'] = electronics.groupby('userId')['Rating'].count()
    st.pyplot(scatterplot('Rating_Mean', 'Rating_Count', data=ratings))

if 'show_buttons' not in st.session_state:
    st.session_state.show_buttons = False

def toggle_buttons():
    st.session_state.show_buttons = not st.session_state.show_buttons

if st.button('View Visualizations', on_click=toggle_buttons):
    pass

if st.session_state.show_buttons:
    #col1, col2, col3 = st.columns(3)

    #with col1:
    if st.button('a. Dataset head()'):
        notebook_cell_1()

    #with col2:
    if st.button('b. Dataset shape'):
        notebook_cell_2()

    #with col3:
    if st.button('c. 5 Point Data Summary'):
        notebook_cell_3()

    if st.button('d. Rating Distribution'):
        notebook_cell_4()

    if st.button('e. Top Rating Count Distribution grouped by Products'):
        notebook_cell_5()

    if st.button('f. Top Rating Count Distribution grouped by Users'):
        notebook_cell_6()

    if st.button('g. Mean Rating Distribution grouped by Products'):
        notebook_cell_7()

    if st.button('h. Mean Rating - Rating Count Distribution grouped by Products'):
        notebook_cell_8()

    if st.button('i. Mean Rating Distribution grouped by Users'):
        notebook_cell_9()

    if st.button('j. Mean Rating - Rating Count Distribution grouped by Users'):
        notebook_cell_10()



def notebook_cell_11():
    class popularity_based_recommender_model():
        def __init__(self, train_data, test_data, user_id, item_id):
            self.train_data = train_data
            self.test_data = test_data
            self.user_id = user_id
            self.item_id = item_id
            self.popularity_recommendations = None

        #Create the popularity based recommender system model
        def fit(self):
            #Get a count of user_ids for each unique product as recommendation score
            train_data_grouped = train_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
            train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)

            #Sort the products based upon recommendation score
            train_data_sort = train_data_grouped.sort_values(['score', self.item_id], ascending = [0,1])

            #Generate a recommendation rank based upon score
            train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')

            #Get the top 10 recommendations
            self.popularity_recommendations = train_data_sort.head(20)

        #Use the popularity based recommender system model to make recommendations
        def recommend(self, user_id, n=5):
            user_recommendations = self.popularity_recommendations

            #Filter products that are not rated by the user
            products_already_rated_by_user = self.train_data[self.train_data[self.user_id] == user_id][self.item_id]
            user_recommendations = user_recommendations[~user_recommendations[self.item_id].isin(products_already_rated_by_user)]

            #Add user_id column for which the recommendations are being generated
            user_recommendations['user_id'] = user_id

            #Bring user_id column to the front
            cols = user_recommendations.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            user_recommendations = user_recommendations[cols].head(n)
            self.plot(user_recommendations)
            return user_recommendations

        def plot(self, user_recommendations):
            f, axes = plt.subplots(1, 2, figsize=(20, 8))
            cplot1 = sns.barplot(x='Rank', y='score', data=user_recommendations, hue='Rank', ax=axes[0])
            cplot1.set_xlabel('Rank',fontsize=20)
            cplot1.set_ylabel('score',fontsize=20)
            cplot2 = sns.pointplot(x='Rank', y='score', data=user_recommendations, hue='Rank', ax=axes[1])
            cplot2.set_xlabel('Rank',fontsize=20)
            cplot2.set_ylabel('score',fontsize=20)

        def predict_evaluate(self):
            ratings = pd.DataFrame(self.train_data.groupby(self.item_id)['Rating'].mean())

            pred_ratings = []
            for data in self.test_data.values:
                if(data[1] in (ratings.index)):
                    pred_ratings.append(ratings.loc[data[1]])
                else:
                    pred_ratings.append(0)

            variable2 = np.asarray(self.test_data['Rating'], dtype="object")
            variable1 = np.asarray(pred_ratings, dtype="object")

            mse = mean_squared_error(variable2, variable1)
            rmse = sqrt(mse)
            return rmse
    pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='userId', item_id='productId')
    pr.fit()
    result_pop_user1 = pr.recommend('ANTN61S4L7WG9')
    st.write(result_pop_user1)
    result_pop_user2 = pr.recommend('AYNAH993VDECT')
    st.write(result_pop_user2)
    result_pop_user3 = pr.recommend('A18YMFFJW974QS')
    st.write(result_pop_user3)
    st.write(display_side_by_side([result_pop_user1, result_pop_user2, result_pop_user3]))

    '''pred_ratings = []
    ratings = pd.DataFrame(train_data.groupby(item_id)['Rating'].mean())
    for data in test_data.values:
        item_id = data[1]
        user_rating = ratings.get(item_id, 0)  # Default to np.nan if item_id is not found
        pred_ratings.append(user_rating)

        st.write("Type of test_data['Rating']:", type(test_data['Rating']))
        st.write("Type of pred_ratings:", type(pred_ratings))

        st.write("Length of test_data['Rating']:", len(test_data['Rating']))
        st.write("Length of pred_ratings:", len(pred_ratings))

        # Let's check the first few entries to see if they are scalar values
        st.write("First few actual ratings:", test_data['Rating'].head())
        st.write("First few predicted ratings:", pred_ratings[:5])

        # Ensure pred_ratings is indeed a flat list
        pred_ratings = np.array(pred_ratings).flatten()
        st.write("Shape of flattened pred_ratings:", pred_ratings.shape)

        # Now let's try calling mean_squared_error again
        mse = mean_squared_error(test_data['Rating'], pred_ratings)'''
    
    st.write(pr.predict_evaluate())

st.markdown("<hr></hr>", unsafe_allow_html=True)

if st.button('Run Popularity Based Recommender Model'):
    notebook_cell_11()