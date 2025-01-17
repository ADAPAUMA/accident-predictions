import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Title and Introduction
st.title("Smart Road Accident Analysis & Prediction")
st.write("""
This application uses machine learning and IoT concepts to analyze road accidents and predict their severity. 
Explore accident data, predict outcomes, and discover actionable insights for safer roads.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Data Preprocessing
    st.write("### Data Cleaning and Preprocessing")
    if st.checkbox("Handle Missing Values"):
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
        df.fillna(method='ffill', inplace=True)
        st.write("Missing values handled successfully.")

    # Feature Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    # Interactive Filters in Sidebar
    st.sidebar.header("Filters")
    severity_filter = st.sidebar.multiselect("Select Accident Severity", df['Accident_severity'].unique())
    cause_filter = st.sidebar.multiselect("Select Cause of Accident", df['Cause_of_accident'].unique())

    # Apply Filters
    filtered_df = df[
        (df['Accident_severity'].isin(severity_filter)) &
        (df['Cause_of_accident'].isin(cause_filter))
    ]
    st.write("### Filtered Dataset Preview")
    st.dataframe(filtered_df.head())

    # Alternative Visualization: Accident Severity Heatmap
    st.write("### Accident Severity Heatmap")
    if "Accident_severity" in df.columns:
        severity_counts = df['Accident_severity'].value_counts().reset_index()
        severity_counts.columns = ['Accident Severity', 'Count']
        fig, ax = plt.subplots()
        sns.heatmap(severity_counts.set_index('Accident Severity').T, annot=True, cmap="YlGnBu", cbar=False, ax=ax)
        st.pyplot(fig)

    # Alternative Visualization: Causes of Accidents as Bar Chart
    st.write("### Causes of Accidents")
    if "Cause_of_accident" in df.columns:
        cause_counts = df['Cause_of_accident'].value_counts()
        fig2, ax2 = plt.subplots()
        sns.barplot(x=cause_counts.values, y=cause_counts.index, orient='h', ax=ax2, palette="viridis")
        ax2.set_title("Accident Causes")
        ax2.set_xlabel("Count")
        ax2.set_ylabel("Cause of Accident")
        st.pyplot(fig2)

    # Additional Visualization: Driving Experience vs Accident Severity (Box Plot)
    st.write("### Driving Experience vs Accident Severity")
    if "Driving_experience" in df.columns and "Accident_severity" in df.columns:
        fig3 = px.box(df, x='Driving_experience', y='Accident_severity', 
                      title='Driving Experience vs Accident Severity')
        st.plotly_chart(fig3)

    # Real-Time Accident Prediction with Random Forest
    st.write("### Predict Accident Severity with Random Forest")
    if st.checkbox("Predict Severity Based on Conditions"):
        # Splitting the data
        X = df.drop('Accident_severity', axis=1)
        y = df['Accident_severity']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the Random Forest model
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Display Model Accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"#### Model Accuracy: {accuracy:.2f}")

        # User Input for Prediction
        st.write("Enter the conditions to predict severity:")
        user_input = {}
        for col in X.columns:
            if df[col].dtype == 'object' or len(df[col].unique()) < 10:
                user_input[col] = st.selectbox(f"Select {col}:", df[col].unique())
            else:
                user_input[col] = st.slider(f"Select {col}:", min_value=int(df[col].min()), max_value=int(df[col].max()), value=int(df[col].mean()))
        
        # Prediction
        user_input_df = pd.DataFrame([user_input])
        prediction = clf.predict(user_input_df)[0]
        st.write(f"### Predicted Accident Severity: {prediction}")

        # Feature Importance Visualization
        st.write("#### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))

    # IoT Predictions
    st.write("### IoT Predictions")
    if st.checkbox("Simulate IoT Accident Prediction"):
        st.write("Using IoT sensors to monitor real-time traffic and environment conditions, predictions can be made about accident likelihood.")
        col1, col2, col3 = st.columns(3)

        with col1:
            traffic_density = np.random.randint(50, 200)
            st.metric(label="Traffic Density (Vehicles/km)", value=traffic_density)

        with col2:
            weather_condition = np.random.choice(["Clear", "Rainy", "Foggy", "Snowy"])
            st.metric(label="Weather Condition", value=weather_condition)

        with col3:
            accident_risk = np.random.choice(["Low", "Medium", "High"])
            st.metric(label="Accident Risk", value=accident_risk)

        st.write("#### IoT-Driven Prediction Result")
        simulated_data = {
            'Traffic Density': traffic_density,
            'Weather Condition': weather_condition,
            'Accident Risk': accident_risk
        }
        st.json(simulated_data)

    # Additional Visualization: Locations with Most Accidents
    st.write("### Locations with Most Accidents")
    if "Location" in df.columns:
        location_counts = df['Location'].value_counts().head(10)
        fig4, ax4 = plt.subplots()
        sns.barplot(x=location_counts.values, y=location_counts.index, ax=ax4, palette="coolwarm")
        ax4.set_title("Top 10 Accident-Prone Locations")
        ax4.set_xlabel("Number of Accidents")
        ax4.set_ylabel("Location")
        st.pyplot(fig4)

    st.write("### Thank you for exploring! Drive safe and stay informed!")
