import streamlit as st
import requests
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Load model for feature importance
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl'))
    return joblib.load(model_path)

# Load accuracy
@st.cache_data
def load_accuracy():
    accuracy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'accuracy.txt'))
    try:
        with open(accuracy_path, 'r') as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return None

def main():
    st.title("🏦 Customer Churn Prediction Dashboard")
    
    # Load model and accuracy
    model = load_model()
    accuracy = load_accuracy()
    
    # Display model accuracy
    if accuracy:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Customer Information")
        
        # Input fields
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")
        salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        # Predict button
        if st.button("🔮 Predict Churn", type="primary"):
            # Prepare data for API
            customer_data = {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": salary,
                "Geography": geography,
                "Gender": gender
            }
            
            try:
                # Send POST request to FastAPI backend
                response = requests.post("http://localhost:8000/predict", json=customer_data)
                response.raise_for_status()
                result = response.json()
                
                # Store result in session state
                st.session_state.prediction = result["prediction"]
                st.session_state.probability = result["churn_probability"]
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error connecting to backend: {str(e)}")
                st.info("💡 Make sure the FastAPI backend is running with: `uvicorn backend.main:app --reload`")
    
    with col2:
        st.header("📊 Prediction Results")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state.prediction
            probability = st.session_state.probability
            
            # Display prediction
            if prediction == 1:
                st.error("🚨 **Churn Predicted**")
                st.write("This customer is likely to churn.")
            else:
                st.success("✅ **No Churn Predicted**")
                st.write("This customer is likely to stay.")
            
            # Display probability
            st.subheader("Churn Probability")
            st.write(f"**{probability:.2%}**")
            
            # Progress bar
            st.progress(probability)
            
            # Gauge-like visualization
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh([0], [probability], color='red', alpha=0.7)
            ax.barh([0], [1-probability], left=[probability], color='green', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0%', '50%', '100%'])
            ax.set_title('Churn Probability')
            st.pyplot(fig)
    
    # Feature Importance Section
    st.header("🔍 Feature Importance")
    try:
        feature_importance = model.named_steps['classifier'].feature_importances_
        feature_names = model.feature_names_in_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        ax.invert_yaxis()
        st.pyplot(fig)
        
        # Display as table
        st.dataframe(importance_df.head(10))
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {str(e)}")

if __name__ == "__main__":
    main()