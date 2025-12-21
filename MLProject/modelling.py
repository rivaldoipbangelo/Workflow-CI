import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/data.csv")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    # MLflow tracking
    mlflow.set_experiment("ML_Training_Experiment")
    
    with mlflow.start_run():
        # Load data
        df = pd.read_csv(args.data_path)
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        
        # Train model
        model = RandomForestClassifier(random_state=args.random_state)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{args.model_name}.pkl"
        joblib.dump(model, model_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path)
        
        print(f"Model accuracy: {accuracy}")
        print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
