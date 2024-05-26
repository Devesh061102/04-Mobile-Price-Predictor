import streamlit as st
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_curve, auc

# Load the dataset
df = pd.read_csv('artifacts\data_ingestion\Mobile-data-train.csv')

# Helper function to display count plot
def display_count_plot(column):
    fig, ax = plt.subplots()
    sns.countplot(x=column, data=df, ax=ax)
    st.pyplot(fig)

# Helper function to display KDE plot
def display_kde_plot(column):
    fig, ax = plt.subplots()
    sns.kdeplot(df[column], ax=ax)
    st.pyplot(fig)

# Helper function to display box plot
def display_box_plot(column):
    fig, ax = plt.subplots()
    sns.boxplot(df[column], ax=ax)
    st.pyplot(fig)


# Sidebar options
st.sidebar.markdown(
    "<h1 style='font-size: 24px;'>Phone Price Range Detector</h1>",
    unsafe_allow_html=True
)

option = st.sidebar.radio("Select an option", ["About", "Dataset", "Analysis", "Model"])

if option == "About":
    st.markdown(
        """
        ## About

        ### Introduction

        The price of a cell phone, a necessity in our daily lives, varies widely depending on its specifications. In this notebook, we will explore the factors affecting cell phone prices and predict new samples based on the best model.

        ### Objective

        Our objective is to predict the price range of a mobile phone by building a model that considers various features provided in the dataset. We will use supervised learning methods such as **Decision Trees (DTs)**, **Random Forest**, and **Support Vector Machine (SVM)** to determine the best model for this problem.

        ### Attributes and Information

        | **No.** | **Attribute**    | **Description** |
        |:-------:|:-----------------|:----------------|
        | **1**   | battery_power    | Total energy a battery can store at one time (mAh) |
        | **2**   | blue             | Has Bluetooth or not |
        | **3**   | clock_speed      | Speed at which microprocessor executes instructions |
        | **4**   | dual_sim         | Has dual SIM support or not |
        | **5**   | fc               | Front camera (Megapixels) |
        | **6**   | four_g           | Has 4G or not |
        | **7**   | int_memory       | Internal memory (Gigabytes) |
        | **8**   | m_dep            | Mobile depth (cm) |
        | **9**   | mobile_wt        | Weight of mobile phone |
        | **10**  | pc               | Primary camera (Megapixels) |
        | **11**  | px_height        | Pixel resolution height |
        | **12**  | px_width         | Pixel resolution width |
        | **13**  | ram              | Random access memory (Megabytes) |
        | **14**  | sc_h             | Screen height (cm) |
        | **15**  | sc_w             | Screen width (cm) |
        | **16**  | talk_time        | Longest time a single battery charge lasts during constant talking |
        | **17**  | three_g          | Has 3G or not |
        | **18**  | touch_screen     | Has touch screen or not |
        | **19**  | wifi             | Has Wi-Fi or not |
        | **20**  | n_cores          | Number of processor cores |
        | **21**  | **price_range**  | Target variable: **0: Low Cost**, **1: Medium Cost**, **2: High Cost**, **3: Very High Cost** |

        ###
        ### Problem Statement

        Our task is to classify the target variable "Price Range" based on the data and attribute information. To achieve the best possible classification, we will develop a model that accurately predicts the price range of mobile phones.
        """,
        unsafe_allow_html=True
    )


elif option == "Dataset":
    # Displaying the title
    st.title("Dataset")

    # Dropdown for preview options
    preview_options = ["First 10 rows", "Last 10 rows", "Random 10 rows", "Whole dataset"]
    selected_option = st.selectbox("Select preview option:", preview_options)

    # Displaying the selected preview
    if selected_option == "First 10 rows":
        st.write("Preview of the first 10 rows:")
        st.write(df.head(10))
    elif selected_option == "Last 10 rows":
        st.write("Preview of the last 10 rows:")
        st.write(df.tail(10))
    elif selected_option == "Random 10 rows":
        st.write("Preview of 10 random rows:")
        st.write(df.sample(10))
    elif selected_option == "Whole dataset":
        st.write("Whole dataset:")
        st.write(df)

    # Displaying the dimensions of the dataset
    st.title("Dimension")
    rows, col = df.shape
    st.write("Dimensions of dataset: ", (rows, col))
    st.write("Rows: ", rows)
    st.write("Columns: ", col)

    # Displaying information about empty values
    st.title("Missing Values")
    empty_values = df.isnull().sum()
    st.write("Number of empty values in each column:")
    st.write(empty_values)

    # Displaying information about duplicate values
    st.title("Duplicate Values")
    duplicate_values = df.duplicated().sum()
    st.write("Number of duplicate rows in the dataset:", duplicate_values)
    # Displaying the information about the dataframe
    st.title("Information")
    info_text = """
    | Column         | Non-Null Count | Dtype   |
    |----------------|----------------|---------|
    | battery_power  | 2000 non-null | int64   |
    | blue           | 2000 non-null | int64   |
    | clock_speed    | 2000 non-null | float64 |
    | dual_sim       | 2000 non-null | int64   |
    | fc             | 2000 non-null | int64   |
    | four_g         | 2000 non-null | int64   |
    | int_memory     | 2000 non-null | int64   |
    | m_dep          | 2000 non-null | float64 |
    | mobile_wt      | 2000 non-null | int64   |
    | n_cores        | 2000 non-null | int64   |
    | pc             | 2000 non-null | int64   |
    | px_height      | 2000 non-null | int64   |
    | px_width       | 2000 non-null | int64   |
    | ram            | 2000 non-null | int64   |
    | sc_h           | 2000 non-null | int64   |
    | sc_w           | 2000 non-null | int64   |
    | talk_time      | 2000 non-null | int64   |
    | three_g        | 2000 non-null | int64   |
    | touch_screen   | 2000 non-null | int64   |
    | wifi           | 2000 non-null | int64   |
    | price_range    | 2000 non-null | int64   |
    """
    st.markdown(info_text, unsafe_allow_html=True)

    # Displaying statistical details with styling
    st.title("Statistical Information")
    st.write(df.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False)\
            .style.background_gradient(cmap="Greens")\
            .bar(subset=["max"], color='#F8766D')\
            .bar(subset=["mean"], color='#00BFC4'))


elif option == "Analysis":
    analysis_option = st.sidebar.selectbox("Select Analysis Type", ["Univariate Analysis", "Bivariate Analysis"])
    
    if analysis_option == "Univariate Analysis":
        st.title("Univariate Analysis")
        analysis_type = st.sidebar.selectbox("Select Data Type", ["Categorical", "Continuous"])
        
        categorical_columns = [col for col in df.columns if df[col].nunique() < 25]
        continuous_columns = [col for col in df.columns if df[col].nunique() >= 25]
        
        if analysis_type == "Categorical":
            st.subheader("Count Plot for Categorical Data")
            selected_column = st.selectbox("Select Column", categorical_columns)
            display_count_plot(selected_column)
        
        elif analysis_type == "Continuous":
            st.subheader("KDE Plot for Continuous Data")
            selected_column_kde = st.selectbox("Select Column for KDE Plot", continuous_columns)
            display_kde_plot(selected_column_kde)

            st.subheader("Box Plot for Continuous Data")
            selected_column_box = st.selectbox("Select Column for Box Plot", continuous_columns)
            display_box_plot(selected_column_box)
    
    elif analysis_option == "Bivariate Analysis":
        st.title("Bivariate Analysis")
        analysis_type = st.sidebar.selectbox("Select Data Type", ["Categorical", "Continuous"])
        
        categorical_columns = [col for col in df.columns if df[col].nunique() < 25]
        
        if analysis_type == "Categorical":
            st.subheader("Count Plot for Categorical Data")
            selected_column1 = st.selectbox("Select Column 1", categorical_columns)
            selected_column2 = st.selectbox("Select Column 2", categorical_columns)
            fig, ax = plt.subplots()
            sns.countplot(x=selected_column1, hue=selected_column2, data=df, ax=ax)
            st.pyplot(fig)

        elif analysis_type == "Continuous":
            st.write("Continuous bivariate analysis is not implemented yet.")

elif option == "Model":
    st.title("Model Training")
    model_option = st.sidebar.selectbox("Select Model", ["SVM", "Decision Tree", "Random Forest", "Naive Bayes"])
    
    if model_option == "SVM":
        C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        gamma = st.sidebar.slider("Gamma", 0.01, 1.0)
        params = {"C": C, "kernel": kernel, "gamma": gamma}
    
    elif model_option == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 1, 32)
        params = {"criterion": criterion, "max_depth": max_depth}
    
    elif model_option == "Random Forest":
        n_estimators = st.sidebar.slider("Number of Trees", 10, 500)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = st.sidebar.slider("Max Depth", 1, 32)
        params = {"n_estimators": n_estimators, "criterion": criterion, "max_depth": max_depth}
    
    elif model_option == "Naive Bayes":
        params = {}
    
    test_size = st.sidebar.selectbox("Test Size (%)", [15, 20, 25, 30, 35]) / 100.0
    
    if st.sidebar.button("Train"):
        if 'price_range' not in df.columns:
            st.write("Error: 'price_range' column not found in dataset.")
        else:
            X = df.drop("price_range", axis=1)
            y = df["price_range"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if model_option == "SVM":
                model = SVC(**params)
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier(**params)
            elif model_option == "Random Forest":
                model = RandomForestClassifier(**params)
            elif model_option == "Naive Bayes":
                model = GaussianNB()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
            
            if model_option != "Naive Bayes":  # ROC curve is generally not used with Naive Bayes
                st.write("### ROC Curve")
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)

            st.write("### Scatter Plot of Predictions")
            fig, ax = plt.subplots()
            ax.scatter(range(len(y_test)), y_test, color='blue', label='Original')
            ax.scatter(range(len(y_test)), y_pred, color='red', alpha=0.5, label='Predicted')
            ax.legend(loc='best')
            st.pyplot(fig)
