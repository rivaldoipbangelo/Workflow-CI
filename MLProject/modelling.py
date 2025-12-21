import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
import os
import glob

def load_dataset():
    """Load dataset from preprocessing folder or generate dummy data"""
    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)
    
    # Cari file CSV di folder preprocessing
    csv_files = glob.glob("spiderman_youtube_review_preprocessing/*.csv")
    
    if csv_files:
        dataset_path = csv_files[0]
        print(f"✓ Found dataset: {dataset_path}")
        try:
            df = pd.read_csv(dataset_path)
            print(f"✓ Dataset loaded successfully: {df.shape}")
            
            # Cek apakah ada kolom target
            if 'target' not in df.columns and 'label' not in df.columns:
                print("⚠ No 'target' or 'label' column found. Using last column as target.")
                target_col = df.columns[-1]
                df = df.rename(columns={target_col: 'target'})
            elif 'label' in df.columns:
                df = df.rename(columns={'label': 'target'})
            
            return df
        except Exception as e:
            print(f"⚠ Error loading dataset: {e}")
            print("⚠ Generating dummy data instead...")
    else:
        print("⚠ No CSV file found in preprocessing folder")
        print("⚠ Generating dummy data...")
    
    # Generate dummy data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    print(f"✓ Dummy dataset generated: {df.shape}")
    
    return df

def main():
    print("\n" + "=" * 60)
    print("MLflow Training Pipeline Started")
    print("=" * 60 + "\n")
    
    # Set MLflow experiment
    mlflow.set_experiment("Spiderman_Review_Classification")
    
    with mlflow.start_run():
        # Load dataset
        df = load_dataset()
        
        # Prepare data
        print("\n" + "=" * 60)
        print("Preparing Data...")
        print("=" * 60)
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle non-numeric columns
        X = X.select_dtypes(include=['int64', 'float64'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Train model
        print("\n" + "=" * 60)
        print("Training Model...")
        print("=" * 60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("✓ Model trained successfully")
        
        # Evaluate
        print("\n" + "=" * 60)
        print("Evaluating Model...")
        print("=" * 60)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Training Accuracy: {train_score:.4f}")
        print(f"✓ Testing Accuracy: {test_score:.4f}")
        print(f"✓ Validation Accuracy: {accuracy:.4f}")
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("n_features", X_train.shape[1])
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model
        print("\n" + "=" * 60)
        print("Saving Model...")
        print("=" * 60)
        
        os.makedirs("models", exist_ok=True)
        model_path = "models/spiderman_review_model.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        print("✓ Model logged to MLflow")
        
        print("\n" + "=" * 60)
        print("✓✓✓ Training Pipeline Completed Successfully! ✓✓✓")
        print("=" * 60 + "\n")
        
        return accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")
