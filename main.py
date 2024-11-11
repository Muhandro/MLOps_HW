import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler,
    MinMaxScaler, RobustScaler
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss
)

st.title("MLOps HW")

# Create Tabs
tabs = st.tabs(["Model Training", "Model List", "Prediction on New Data", "Model Evaluation"])

with tabs[0]:
    st.header("Model Training")

    # **Data Upload**
    uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head())

        # **Column Display with Data Types**
        st.subheader("Data Types")
        data_types = pd.DataFrame(data.dtypes, columns=['Type']).reset_index()
        data_types.columns = ['Column', 'Type']
        st.table(data_types)

        # **Target Column Selection**
        target_column = st.selectbox("Select the target column", options=data.columns)

        # **Preprocessing Selection**
        st.subheader("Preprocessing Options")
        preprocessing_methods = {}
        for column in data.columns:
            if column != target_column:
                col_type = data[column].dtype
                st.write(f"**{column}** ({col_type})")
                if col_type == 'object':
                    method = st.selectbox(
                        f"Select preprocessing for '{column}'",
                        [
                            'Label Encoding',
                            'One-Hot Encoding',
                            'Target Encoding',
                            'Frequency Encoding',
                            'Binary Encoding',
                            'Hashing Encoding'
                        ],
                        key=f"preprocess_{column}"
                    )
                elif np.issubdtype(col_type, np.integer):
                    method = st.selectbox(
                        f"Select preprocessing for '{column}'",
                        [
                            'None',
                            'Standard Scaling',
                            'Min-Max Scaling',
                            'Robust Scaling'
                        ],
                        key=f"preprocess_{column}"
                    )
                else:
                    method = 'None'
                preprocessing_methods[column] = method

        # **Model Selection**
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Choose a model to train",
            options=['Random Forest', 'Gradient Boosting']
        )

        # **Hyperparameter Adjustment**
        st.subheader("Hyperparameter Tuning")
        hyperparameters = {}
        if model_choice == 'Random Forest':
            hyperparameters['n_estimators'] = st.slider("Number of Trees (n_estimators)", 10, 500, 100)
            hyperparameters['max_depth'] = st.slider("Maximum Depth (max_depth)", 1, 50, 10)
            hyperparameters['min_samples_split'] = st.slider("Min Samples Split (min_samples_split)", 2, 10, 2)
            hyperparameters['min_samples_leaf'] = st.slider("Min Samples Leaf (min_samples_leaf)", 1, 10, 1)
            hyperparameters['max_features'] = st.selectbox("Max Features (max_features)", options=['auto', 'sqrt', 'log2'])
            hyperparameters['bootstrap'] = st.checkbox("Bootstrap Samples", value=True)
        elif model_choice == 'Gradient Boosting':
            hyperparameters['n_estimators'] = st.slider("Number of Trees (n_estimators)", 10, 500, 100)
            hyperparameters['learning_rate'] = st.number_input("Learning Rate (learning_rate)", 0.001, 1.0, 0.1)
            hyperparameters['max_depth'] = st.slider("Maximum Depth (max_depth)", 1, 50, 3)
            hyperparameters['min_samples_split'] = st.slider("Min Samples Split (min_samples_split)", 2, 10, 2)
            hyperparameters['min_samples_leaf'] = st.slider("Min Samples Leaf (min_samples_leaf)", 1, 10, 1)
            hyperparameters['subsample'] = st.number_input("Subsample (subsample)", 0.1, 1.0, 1.0)

        # **Train/Test Split**
        st.subheader("Train/Test Split")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100.0

        # **Metric Selection**
        st.subheader("Select Evaluation Metrics")
        metrics_options = [
            'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC',
            'Confusion Matrix', 'Matthews Correlation Coefficient',
            'Cohen Kappa Score', 'Log Loss', 'Brier Score Loss'
        ]
        selected_metrics = st.multiselect("Choose metrics to evaluate", options=metrics_options)

        # **Training Button**
        if st.button("Train Model"):
            # **Data Preprocessing**
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Apply preprocessing methods
            for col, method in preprocessing_methods.items():
                if method == 'Label Encoding':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                elif method == 'One-Hot Encoding':
                    X = pd.get_dummies(X, columns=[col])
                elif method == 'Target Encoding':
                    target_mean = data.groupby(col)[target_column].mean()
                    X[col] = X[col].map(target_mean)
                elif method == 'Frequency Encoding':
                    freq = data[col].value_counts() / len(data)
                    X[col] = X[col].map(freq)
                elif method == 'Binary Encoding':
                    X[col] = X[col].apply(lambda x: ''.join(format(ord(c), 'b') for c in str(x)))
                elif method == 'Hashing Encoding':
                    X[col] = X[col].apply(lambda x: hash(str(x)) % 5000)
                elif method == 'Standard Scaling':
                    scaler = StandardScaler()
                    X[col] = scaler.fit_transform(X[[col]])
                elif method == 'Min-Max Scaling':
                    scaler = MinMaxScaler()
                    X[col] = scaler.fit_transform(X[[col]])
                elif method == 'Robust Scaling':
                    scaler = RobustScaler()
                    X[col] = scaler.fit_transform(X[[col]])

            # **Train/Test Split**
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # **Model Training**
            if model_choice == 'Random Forest':
                model = RandomForestClassifier(**hyperparameters)
            elif model_choice == 'Gradient Boosting':
                model = GradientBoostingClassifier(**hyperparameters)
            model.fit(X_train, y_train)

            # **Model Evaluation**
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            st.subheader("Model Evaluation Results")
            for metric in selected_metrics:
                if metric == 'Accuracy':
                    score = accuracy_score(y_test, y_pred)
                    st.write(f"**Accuracy**: {score:.4f}")
                elif metric == 'Precision':
                    score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    st.write(f"**Precision**: {score:.4f}")
                elif metric == 'Recall':
                    score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    st.write(f"**Recall**: {score:.4f}")
                elif metric == 'F1 Score':
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    st.write(f"**F1 Score**: {score:.4f}")
                elif metric == 'ROC AUC' and y_prob is not None:
                    score = roc_auc_score(y_test, y_prob)
                    st.write(f"**ROC AUC**: {score:.4f}")
                elif metric == 'Confusion Matrix':
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("**Confusion Matrix**:")
                    st.write(cm)
                elif metric == 'Matthews Correlation Coefficient':
                    score = matthews_corrcoef(y_test, y_pred)
                    st.write(f"**Matthews Corrcoef**: {score:.4f}")
                elif metric == 'Cohen Kappa Score':
                    score = cohen_kappa_score(y_test, y_pred)
                    st.write(f"**Cohen Kappa Score**: {score:.4f}")
                elif metric == 'Log Loss' and y_prob is not None:
                    score = log_loss(y_test, y_prob)
                    st.write(f"**Log Loss**: {score:.4f}")
                elif metric == 'Brier Score Loss' and y_prob is not None:
                    score = brier_score_loss(y_test, y_prob)
                    st.write(f"**Brier Score Loss**: {score:.4f}")
                else:
                    st.write(f"Metric '{metric}' is not applicable.")


with tabs[1]:
    st.header("Model List")
    st.write("List of trained models will be displayed here.")


with tabs[2]:
    st.header("Prediction on New Data")

    # **Data Upload**
    new_data_file = st.file_uploader("Upload new data for prediction", type=["csv"], key="new_data")
    if new_data_file:
        new_data = pd.read_csv(new_data_file)
        st.write("New Data Preview:")
        st.dataframe(new_data.head())

with tabs[3]:
    st.header("Model Evaluation")
    st.write("Evaluate and compare different models.")

