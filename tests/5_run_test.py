import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

def test_model_load():
    model = mlflow.sklearn.load_model("models:/HeartDiseaseModel/latest")
    assert model is not None

def test_prediction_shape():
    model = mlflow.sklearn.load_model("models:/HeartDiseaseModel/latest")
    sample = pd.DataFrame([{
        "age": 55, "sex": 1, "cp": 2, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2
    }])
    pred = model.predict(sample)
    
    # Print original input and prediction
    print("Original Input Data:")
    print(sample)
    print("Prediction:")
    print(pred)

    assert len(pred) == 1