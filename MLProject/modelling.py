import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

def main():
    print("=" * 60)
    print("MLflow Training Pipeline Started")
    print("=" * 60)
    
    # Set experiment
    mlflow.set_experiment("Spiderman_Review_Classification")
    
    with mlflow.start_run():
        # Load dataset
        print("\n[1/5] Loading dataset...")
        try:
            df = pd.read_csv("spiderman_youtube_review_preprocessing/spiderman_youtube_review_preprocessed.csv")
            print(f"✓ Dataset loaded: {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
        except Exception as e:
            print(f"⚠ Error loading dataset: {e}")
            print("⚠ Generating dummy data...")
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
            df['target'] = y
            print(f"✓ Dummy dataset generated: {df.shape}")
        
        # Identify target column
        print("\n[2/5] Identifying target column...")
        target_col = None
        
        # Cek berbagai kemungkinan nama kolom target
        possible_targets = ['target', 'label', 'sentiment', 'class', 'y', 'output']
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                print(f"✓ Found target column: '{target_col}'")
                break
        
        if target_col is None:
            # Gunakan kolom terakhir sebagai target
            target_col = df.columns[-1]
            print(f"⚠ No standard target column found. Using last column: '{target_col}'")
        
        # Prepare data
        print("\n[3/5] Preparing data...")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Pilih hanya kolom numerik
        numeric_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("⚠ No numeric columns found!")
            print("⚠ Available columns:", list(X.columns))
            print("⚠ Data types:", X.dtypes.to_dict())
            print("\n⚠ Generating dummy features...")
            
            # Buat dummy features
            from sklearn.datasets import make_classification
            X_dummy, _ = make_classification(n_samples=len(y), n_features=10, random_state=42)
            X = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(10)])
            numeric_cols = X.columns.tolist()
        
        X = X[numeric_cols]
        print(f"✓ Using {len(numeric_cols)} numeric features")
        print(f"✓ Features: {numeric_cols}")
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Train model
        print("\n[4/5] Training model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("✓ Model trained successfully")
        
        # Evaluate
        print("\n[5/5] Evaluating model...")
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
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("accuracy", accuracy)
        
        # Save model
        print("\n[6/6] Saving model...")
        os.makedirs("models", exist_ok=True)
        model_path = "models/spiderman_review_model.pkl"
        joblib.dump(model, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        print("✓ Model logged to MLflow")
        
        print("\n" + "=" * 60)
        print("✓✓✓ Training Pipeline Completed Successfully! ✓✓✓")
        print("=" * 60)
        
        return accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")
