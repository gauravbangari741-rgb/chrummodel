import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data():
    """Load the dataset from CSV file."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Churn_Modelling.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    """Preprocess the data: drop unnecessary columns and encode categoricals."""
    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def train_model():
    """Train the machine learning model using a pipeline."""
    # Load and preprocess data
    df = load_data()
    df_processed = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Exited', axis=1)
    y = df_processed['Exited']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    return pipeline, accuracy

if __name__ == "__main__":
    # Train the model
    model, accuracy = train_model()
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    joblib.dump(model, model_path)
    
    # Save accuracy for display
    accuracy_path = os.path.join(os.path.dirname(__file__), 'accuracy.txt')
    with open(accuracy_path, 'w') as f:
        f.write(f"{accuracy:.4f}")
    
    print("Model trained and saved successfully!")