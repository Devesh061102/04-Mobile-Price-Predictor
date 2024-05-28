import streamlit as st
import io
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('artifacts\data_ingestion\Mobile-data-train.csv')

def display_Count_plot(Df, selected_column):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=Df, y=selected_column, palette='Greens', orient='h')
    plt.title(f'Count of {selected_column}', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel(selected_column, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Function to display the count plot per price range using Seaborn
def display_count_plot_per_price_range(Df, selected_column):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=Df, y=selected_column, palette='mako', orient='h', hue='price_range')
    plt.title(f'Count of {selected_column} per Price Range', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel(selected_column, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(plt)

# Function to display the distribution plot using Seaborn
def display_distribution_plot(Df, selected_column):
    plt.figure(figsize=(10, 6))
    sns.set_style('darkgrid')
    sns.kdeplot(data=Df, x=selected_column, hue='price_range', fill=True, palette='Greens')
    plt.title(f'Distribution of {selected_column}', fontsize=16)
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

# Function to display the box plot using Seaborn
def display_box_plot(Df, selected_column):
    plt.figure(figsize=(10, 6))
    sns.set_style('darkgrid')
    sns.boxplot(data=Df, x=selected_column, y='price_range', palette='light:#5A9', orient='h')
    plt.title(f'Box Plot of {selected_column}', fontsize=16)
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel('Price Range', fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

def display_contingency_table(Df, selected_column, selected_column2):
    # Create a contingency table
    contingency_table = pd.crosstab(index=Df[selected_column], columns=Df[selected_column2])

    # Display the contingency table
    st.write(contingency_table)

def display_heatmap_plot(Df, selected_column, selected_column2):
    # Create a contingency table
    contingency_table = pd.crosstab(index=Df[selected_column], columns=Df[selected_column2])

    # Create and display the heatmap plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu", fmt="d")
    plt.title(f'Heatmap Plot of {selected_column} vs {selected_column2}')
    plt.xlabel(selected_column2)
    plt.ylabel(selected_column)
    st.pyplot(plt)

def display_scatter_plot(Df, selected_column, selected_column2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=Df, x=selected_column, y=selected_column2,hue='price_range')
    plt.title(f'Scatter Plot of {selected_column} vs {selected_column2}', fontsize=16)
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel(selected_column2, fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

# Function to display the joint plot
def display_joint_plot(Df, selected_column, selected_column2):
    plt.figure(figsize=(10, 6))
    sns.jointplot(data=Df, x=selected_column, y=selected_column2, kind='hex')
    plt.xlabel(selected_column, fontsize=14)
    plt.ylabel(selected_column2, fontsize=14)
    plt.tight_layout()
    st.pyplot(plt)

def metrics_calculator(y_test, y_pred, model_name):
    '''
    This function calculates all desired performance metrics for a given model.
    '''
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result

# ROC Curve Plot
def roc_curve_plot(y_actual, y_predicted_probs, figsize=(5, 4), title=None, legend_loc='best'):
                fpr = {}
                tpr = {}
                thres = {}
                roc_auc = {}

                n_class = y_predicted_probs.shape[1]
                for i in range(n_class):
                    fpr[i], tpr[i], thres[i] = roc_curve(y_actual == i, y_predicted_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                plt.figure(figsize=figsize)
                for i in range(n_class):
                    plt.plot(fpr[i], tpr[i], linewidth=1, label='Class {}: AUC={:.2f}'.format(i, roc_auc[i]))

                plt.plot([0, 1], [0, 1], '--', linewidth=0.5)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                if title is not None:
                    plt.title(title)
                plt.legend(loc=legend_loc)
                st.pyplot(plt)

# Sidebar options
st.sidebar.markdown(
    "<h1 style='font-size: 24px;'>Phone Price Range Detector</h1>",
    unsafe_allow_html=True
)

option = st.sidebar.radio("Select an option", ["About", "Dataset", "Analysis", "Model","Prediction"])

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


if option == "Analysis":
    analysis_option = st.sidebar.selectbox("Select Analysis Type", ["Univariate Analysis", "Bivariate Analysis"])
    
    if analysis_option == "Univariate Analysis":
        st.title("Univariate Analysis")
        analysis_type = st.sidebar.selectbox("Select Data Type", ["Categorical", "Continuous"])
        
        categorical_columns = [col for col in df.columns if df[col].nunique() < 25]
        continuous_columns = [col for col in df.columns if df[col].nunique() >= 25]
        
        if analysis_type == "Categorical":
            st.subheader("Count plot and Count Chart for Categorical Data")
            selected_column = st.selectbox("Select Column", categorical_columns)
            
            st.subheader("Count Plot")
            display_Count_plot(df, selected_column)
            
            st.subheader("Count Plot per Price Range")
            display_count_plot_per_price_range(df, selected_column)
        
        elif analysis_type == "Continuous":
            st.subheader("Plots for Continuous Data")
            selected_column = st.selectbox("Select Column for Plots", continuous_columns)
            
            st.subheader("Distribution Plot")
            display_distribution_plot(df, selected_column)

            st.subheader("Box Plot")
            display_box_plot(df, selected_column)
            

    
    elif analysis_option == "Bivariate Analysis":
        st.title("Bivariate Analysis")
        analysis_type = st.sidebar.selectbox("Select Data Type", ["Categorical", "Continuous"])
        
        categorical_columns = [col for col in df.columns if df[col].nunique() < 25]
        continuous_columns = [col for col in df.columns if df[col].nunique() >= 25]
        
        if analysis_type == "Categorical":
            st.subheader("Contingency Table and Heatmap Plot")
            selected_column = st.selectbox("Select Column", categorical_columns)
            selected_column2 = st.selectbox("Select Column2", categorical_columns)

            st.subheader("Contingency Table")
            display_contingency_table(df, selected_column, selected_column2)

            st.subheader("Heatmap Plot")
            display_heatmap_plot(df, selected_column, selected_column2)

            
        elif analysis_type == "Continuous":
            st.subheader("Scatter Plot and Joint Plot")
            selected_column = st.selectbox("Select Column", continuous_columns)
            selected_column2 = st.selectbox("Select Column2", continuous_columns)

            st.subheader("Scatter Plot")
            display_scatter_plot(df, selected_column, selected_column2)

            st.subheader("Joint Plot")
            display_joint_plot(df, selected_column, selected_column2)


elif option == "Model":
    st.title("Model Training")
    model_option = st.sidebar.selectbox("Select Model", ["SVM", "Decision Tree", "Random Forest"])
    
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
    
    test_size = st.sidebar.selectbox("Test Size (%)", [15, 20, 25, 30, 35]) / 100.0
    
    if st.sidebar.button("Train"):
        if 'price_range' not in df.columns:
            st.write("Error: 'price_range' column not found in dataset.")
        else:
            X = df.drop("price_range", axis=1)
            y = df["price_range"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            if model_option == "SVM":
                model = SVC(**params, probability=True)
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier(**params)
            elif model_option == "Random Forest":
                model = RandomForestClassifier(**params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Confusion Matrix Heatmap
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.write("### Performace Matrix")
            BaseRF_result = metrics_calculator(y_test, y_pred, model)
            BaseRF_result

            # ROC Curve
            
            st.write("### ROC Curve")
            y_pred_prob = model.predict_proba(X_test)
            roc_curve_plot(y_test, y_pred_prob, title='ROC Curve for ' + model_option)

            st.session_state['model'] = model

elif option == "Prediction":
    st.title("Predict Mobile Price Range")

    # Create a form for user input
    with st.form(key='prediction_form'):
        st.write("Enter the values for the features to predict the price range:")

        # Define input fields for each feature
        battery_power = st.number_input("Battery Power (500 - 2000)", min_value=500, max_value=2000)
        blue = st.selectbox("Bluetooth", ["No", "Yes"])
        clock_speed = st.number_input("Clock Speed (0.5 - 3.0)", min_value=0.5, max_value=3.0, format="%.2f")
        dual_sim = st.selectbox("Dual SIM", ["No", "Yes"])
        fc = st.number_input("Front Camera (0 - 19 Megapixels)", min_value=0, max_value=19)
        four_g = st.selectbox("4G", ["No", "Yes"])
        int_memory = st.number_input("Internal Memory (2 - 64 GB)", min_value=2, max_value=64)
        m_dep = st.number_input("Mobile Depth (0.1 - 1.0 cm)", min_value=0.1, max_value=1.0, format="%.2f")
        mobile_wt = st.number_input("Mobile Weight (80 - 200 grams)", min_value=80, max_value=200)
        n_cores = st.number_input("Number of Cores (1 - 8)", min_value=1, max_value=8)
        pc = st.number_input("Primary Camera (0 - 20 Megapixels)", min_value=0, max_value=20)
        px_height = st.number_input("Pixel Resolution Height (0 - 1960)", min_value=0, max_value=1960)
        px_width = st.number_input("Pixel Resolution Width (500 - 1998)", min_value=500, max_value=1998)
        ram = st.number_input("RAM (256 - 3998 MB)", min_value=256, max_value=3998)
        sc_h = st.number_input("Screen Height (5 - 19 cm)", min_value=5, max_value=19)
        sc_w = st.number_input("Screen Width (0 - 18 cm)", min_value=0, max_value=18)
        talk_time = st.number_input("Talk Time (2 - 20 hours)", min_value=2, max_value=20)
        three_g = st.selectbox("3G", ["No", "Yes"])
        touch_screen = st.selectbox("Touch Screen", ["No", "Yes"])
        wifi = st.selectbox("Wi-Fi", ["No", "Yes"])

        # Submit button
        submit_button = st.form_submit_button(label='Predict')

    # Perform prediction using the trained model
    if submit_button:
        # Convert categorical inputs to numerical values
        blue = 1 if blue == "Yes" else 0
        dual_sim = 1 if dual_sim == "Yes" else 0
        four_g = 1 if four_g == "Yes" else 0
        three_g = 1 if three_g == "Yes" else 0
        touch_screen = 1 if touch_screen == "Yes" else 0
        wifi = 1 if wifi == "Yes" else 0

        input_data = pd.DataFrame({
            'battery_power': [battery_power],
            'blue': [blue],
            'clock_speed': [clock_speed],
            'dual_sim': [dual_sim],
            'fc': [fc],
            'four_g': [four_g],
            'int_memory': [int_memory],
            'm_dep': [m_dep],
            'mobile_wt': [mobile_wt],
            'n_cores': [n_cores],
            'pc': [pc],
            'px_height': [px_height],
            'px_width': [px_width],
            'ram': [ram],
            'sc_h': [sc_h],
            'sc_w': [sc_w],
            'talk_time': [talk_time],
            'three_g': [three_g],
            'touch_screen': [touch_screen],
            'wifi': [wifi]
        })

        # Use the trained model to predict the price range
        if 'model' in st.session_state:
            model = st.session_state['model']
            prediction = model.predict(input_data)[0]
            price_range_mapping = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
            }
            predicted_label = price_range_mapping[prediction]
            st.markdown(f"<h2>Predicted Price Range</h2>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: green;'> {predicted_label}</h2>", unsafe_allow_html=True)
        else:
            st.write("Model not trained yet. Please train the model in the 'Model' section.")
