import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

def app():
    st.title('Graph Data')

    st.write("This is the `Graph Data` page of the multi-page app.")

    st.write("Now let's visualize the data with a scatter plot:")

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    Y = pd.Series(iris.target, name='class')
    df = pd.concat([X, Y], axis=1)
    df['class'] = df['class'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

    # Define colors for each class
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

    # Plot
    fig, ax = plt.subplots()
    for species in df['class'].unique():
        df_species = df[df['class'] == species]
        ax.scatter(df_species['sepal length (cm)'], df_species['sepal width (cm)'],
                   c=colors[species], label=species)

    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title('Sepal Length vs. Sepal Width')
    ax.legend()
    st.pyplot(fig)
