import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns # For mathematical calculations

import matplotlib.pyplot as plt  # For plotting graphs

import streamlit as st
import time



# Define a function to load data
@st.cache_data()
def load_data(file_path):
    return pd.read_csv(file_path)

# Define a function to merge datasets
def merge_datasets(df1, df2):
    # merge the two datasets on a common column
    merged_df = pd.merge(df1, df2, on='common_column')
    return merged_df





# Define a function to merge two datasets
def merge_datasets(data1, data2):
    merged_data = pd.merge(data1, data2, on='key_column')
    return merged_data
# Define function for converting categorical columns
def convert_categorical(df, method):
    if method == 'OneHot':
        encoder = OneHotEncoder()
        encoded_df = pd.DataFrame(encoder.fit_transform(df).toarray())
    elif method == 'Label':
        encoder = LabelEncoder()
        encoded_df = df.apply(encoder.fit_transform)
    return encoded_df
# Define the Streamlit app
def app():
    # Add menu items to the Streamlit app's menu
    
    # Set the page title and icon
    st.set_page_config(page_title='DecisionTree Regression App', page_icon=':memo:', layout='wide',initial_sidebar_state="auto")

    
    # Add a sidebar with a menu
    st.sidebar.write("MENU")
    option = st.sidebar.selectbox(
        "Select an option",
        ("Home", "About Decision Tree Regressor", "For more info")
    )

    # Display different content depending on the selected option
    if option == "Home":
        st.write("Welcome to the home page!")
    elif option == "About Decision Tree Regressor":
        st.write("""Decision trees are one of the most popular algorithms used for machine learning tasks, especially when it comes to regression problems. In this blog, we'll explore what decision tree regressors are and how they work.

What is a Decision Tree Regressor?

A decision tree regressor is a type of regression algorithm that uses a decision tree as a predictive model. It works by recursively splitting the data into smaller subsets based on the value of a particular feature. Each split divides the data into two or more subsets, and the process is repeated until a stopping criterion is met. This produces a tree-like structure where each internal node represents a decision based on a feature, and each leaf node represents a prediction.

How does it work?

The decision tree regressor starts by selecting the feature that provides the best split. The best split is the one that maximizes the difference between the predicted values of the two subsets created by the split. This process is repeated until a stopping criterion is met, such as when all the data in a subset has the same value or when a maximum depth is reached.

To make a prediction, the decision tree regressor starts at the root node of the tree and follows the path through the tree based on the value of the feature at each internal node. The final prediction is the value of the leaf node that is reached.

Advantages of Decision Tree Regressors

There are several advantages to using decision tree regressors:

Easy to interpret: Decision trees are easy to visualize, and the rules used to make predictions are easy to understand.

Non-parametric: Decision trees don't make any assumptions about the distribution of the data, making them useful for datasets with complex relationships.

Robust: Decision trees can handle missing data, outliers, and noisy data.

Can handle both categorical and numerical data: Decision trees can handle both types of data, making them versatile.

Disadvantages of Decision Tree Regressors

There are also some disadvantages to using decision tree regressors:

Overfitting: Decision trees can be prone to overfitting, which can lead to poor performance on new data.

Instability: Small changes in the data can lead to large changes in the structure of the decision tree.

Bias: Decision trees can be biased towards features with many values or towards features that appear earlier in the tree.

Conclusion

Decision tree regressors are a powerful tool for regression tasks. They are easy to interpret and can handle both categorical and numerical data. However, they can be prone to overfitting and instability, so it's important to tune the hyperparameters and use techniques like pruning to prevent these issues. Overall, decision tree regressors are a useful addition to any data scientists toolkit.""")
    elif option == "For more info":
        st.write("follow this link : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html")

    # Set the app title
    st.title('DecisionTree Regression App')
        
    # Allow users to upload two datasets
    st.sidebar.header('Upload datasets')
    uploaded_file1 = st.sidebar.file_uploader('Upload dataset 1', type=['csv', 'xlsx'])
    uploaded_file2 = st.sidebar.file_uploader('Upload dataset 2', type=['csv', 'xlsx'])
  
    # Merge the two datasets
    if uploaded_file1 is not None and uploaded_file2 is not None:
          with st.spinner('Loading datasets...'):
            df1 = load_data(uploaded_file1)
            df2 = load_data(uploaded_file2)
          with st.spinner('Merging datasets...'):
            merged_df = pd.merge(df1, df2, on='id')
            st.write('Merged Dataset')
            st.write(merged_df)



            # Check summary statistics and data types
            st.header('Summary Statistics and Data Types')
            st.write('Summary Statistics')
            st.write(merged_df.describe())
            st.write('Data Types')
            st.write(merged_df.dtypes)

            # Check correlation
            st.header('Correlation')
            st.write(merged_df.corr())


            # Print numerical and categorical features
            st.header('Numerical and Categorical Features')
            numerical_features = merged_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = merged_df.select_dtypes(include=['object']).columns.tolist()
            st.write('Numerical Features')
            st.write(numerical_features)
            st.write('Categorical Features')
            st.write(categorical_features)

            # Perform exploratory data analysis
            st.sidebar.title("Exploratory Data Analysis")
            # get user input for the column to plot
            plot_column = st.sidebar.selectbox('Select a column to plot(HIATOGRAM,COUNTPLOT)', merged_df.columns)
            
            # plot a histogram of the selected column
            plt.figure()
            sns.histplot(data=merged_df, x=plot_column)
            st.pyplot()
            
            # plot a bar chart of the selected column
            if merged_df[plot_column].dtype == 'object':
                plt.figure()
                sns.countplot(data=merged_df, x=plot_column)
                st.pyplot()
            # plot a scatter plot of two numerical columns
            elif merged_df[plot_column].dtype in ['float64', 'int64']:
                plt.figure()
                numerical_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
                x_column = st.sidebar.selectbox('Select a column for X-axis[SCATER PLOT]', numerical_columns)
                y_column = st.sidebar.selectbox('Select a column for Y-axis[SCATER PLOT]', numerical_columns)
                sns.scatterplot(data=merged_df, x=x_column, y=y_column)
                st.pyplot()

            
            
            st.sidebar.subheader("Select column for checking target variable distribution with heatmap")
            target_col = st.sidebar.selectbox("Target variable", merged_df.columns)
            if target_col != 'id':
                st.write("Target variable:", target_col)
                st.write("Summary statistics:")
                st.write(merged_df[target_col].describe())
                st.write("Distribution:")
                sns.histplot(data=merged_df, x=target_col, kde=True)
                st.pyplot()
                st.write("Correlation with other variables:")
                corr = merged_df.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                st.pyplot()

          

        

            # Display outliers in boxplot
            st.header("Display Outliers")
            num_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            col = st.selectbox("Select a numerical column to display outliers", num_cols)
            if col is not None:
                fig, ax = plt.subplots()
                ax.boxplot(merged_df[col], vert=False)
                ax.set_title(f"Boxplot of {col}")
                st.pyplot(fig)

            # Choose outlier treatment method
            outlier_method = st.selectbox("Select outlier treatment method:", ['None', 'Z-score', 'IQR'])

            if outlier_method != 'None':
                # Detect and treat outliers using z-score
                if outlier_method == 'Z-score':
                    z = np.abs(stats.zscore(merged_df.select_dtypes(include=[np.number])))
                    threshold = st.number_input("Enter Z-score threshold value:")
                    merged_df = merged_df[(z < threshold).all(axis=1)]
                    st.write("Outliers removed using Z-score method:")
                    st.write(merged_df)

                # Detect and treat outliers using IQR
                elif outlier_method == 'IQR':
                    q1 = merged_df.select_dtypes(include=[np.number]).quantile(0.25)
                    q3 = merged_df.select_dtypes(include=[np.number]).quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    merged_df = merged_df[~((merged_df.select_dtypes(include=[np.number]) < lower_bound) | (merged_df.select_dtypes(include=[np.number]) > upper_bound)).any(axis=1)]
                    st.write("Outliers removed using IQR method:")
                    st.write(merged_df)


            # Convert categorical columns
            st.sidebar.subheader('Convert Categorical Columns')
            cat_cols = merged_df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                method = st.sidebar.radio('Select conversion method:', ['None','OneHot', 'Label'])
                encoded_df = convert_categorical(merged_df[cat_cols], method)
                merged_df = pd.concat([merged_df.drop(cat_cols, axis=1), encoded_df], axis=1)
                st.write('Converted categorical columns:', cat_cols)
            else:
                st.write('No categorical columns found.')
        
            # Missing value imputation
            st.sidebar.header("Missing Value Imputation")
            strategy = st.sidebar.selectbox("Select missing value imputation strategy",
                                    options=["None","Drop NA", "Mean Imputation", "Median Imputation", "Mode Imputation"])
            if strategy == "Drop NA":
                merged_df.dropna(inplace=True)
            elif strategy == "Mean Imputation":
                merged_df.fillna(merged_df.mean(), inplace=True)
            elif strategy == "Median Imputation":
                merged_df.fillna(merged_df.median(), inplace=True)
            elif strategy == "Mode Imputation":
                merged_df.fillna(merged_df.mode().iloc[0], inplace=True)
            # Add a button to start feature scaling
            st.sidebar.header(" Feature Scaling")
            if st.sidebar.button("Perform Feature Scaling"):
                # Choose a feature scaling method
                scaling_method = st.sidebar.selectbox("Select a scaling method", ["None", "Standardization", "Normalization"])

                # Perform feature scaling
                if scaling_method == "Standardization":
                    scaler = StandardScaler()
                    merged_df = scaler.fit_transform(merged_df)
                elif scaling_method == "Normalization":
                    scaler = MinMaxScaler()
                    merged_df = scaler.fit_transform(merged_df)
                else:
                    merged_df = merged_df


            with st.spinner('Splitting dataset into training and testing sets...'):

                # Allow users to select the target variable and the features for the model
                st.header('Select target variable and features')
                target_col = st.selectbox('Select the target variable', options=merged_df.columns)
                feature_cols = st.multiselect('Select the features', options=merged_df.columns.drop(target_col))

                # Perform train_test_split
                st.header('Train-test split')
                test_size = st.slider('Select the test size', min_value=0.1, max_value=0.5, step=0.1, value=0.2)
                X_train, X_test, y_train, y_test = train_test_split(merged_df[feature_cols], merged_df[target_col], test_size=test_size, random_state=0)
            # train the decision tree regression model
            with st.spinner('Training the model...'):

                # Build the DecisionTree regression model
                st.header('Build model')
                max_depth = st.slider('Select the maximum depth', min_value=1, max_value=10, value=3)
                model = DecisionTreeRegressor(max_depth=max_depth)
                model.fit(X_train, y_train)

                # Evaluate the model performance
                st.header('Model performance')
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write('MSE:', mse)
                st.write('R2 score:', r2)

    'Starting a long computation...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(10):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'





if __name__ == '__main__':
    app()
