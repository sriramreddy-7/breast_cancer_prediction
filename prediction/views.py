import joblib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse

# Load pre-trained model and preprocessors
model = joblib.load(r"D:\Web Development\[02] Django Project's\BCP\BreastCancerPrediction\prediction\rfc.pkl")
scaler = joblib.load(r"D:\Web Development\[02] Django Project's\BCP\BreastCancerPrediction\prediction\scaler.pkl")
encoder = joblib.load(r"D:\Web Development\[02] Django Project's\BCP\BreastCancerPrediction\prediction\encoder.pkl")
print("Model, scaler, and encoder loaded successfully")

def predict_status(request):
    if request.method == "POST":
        try:
            # Extract form data
            form_data = {
                "Age": float(request.POST.get("Age")),
                "Gender": request.POST.get("Gender"),
                "Protein1": float(request.POST.get("Protein1")),
                "Protein2": float(request.POST.get("Protein2")),
                "Protein3": float(request.POST.get("Protein3")),
                "Protein4": float(request.POST.get("Protein4")),
                "Tumour_Stage": request.POST.get("Tumour_Stage"),
                "Histology": request.POST.get("Histology"),
                "ER_status": request.POST.get("ER_status"),
                "PR_status": request.POST.get("PR_status"),
                "HER2_status": request.POST.get("HER2_status"),
                "Surgery_type": request.POST.get("Surgery_type"),
            }

            # Preprocess categorical data (using transform instead of fit_transform)
            categorical_cols = [
                "Gender", "Tumour_Stage", "Histology",
                "ER_status", "PR_status", "HER2_status", "Surgery_type"
            ]
            categorical_values = np.array([encoder.transform([form_data[col]])[0] for col in categorical_cols]).reshape(1, -1)

            # Scale numerical data
            numerical_cols = ["Age", "Protein1", "Protein2", "Protein3", "Protein4"]
            numerical_values = np.array([form_data[col] for col in numerical_cols]).reshape(1, -1)
            scaled_numerical_values = scaler.transform(numerical_values)

            # Combine numerical and categorical data
            input_data = np.hstack([scaled_numerical_values, categorical_values])

            # Make prediction
            prediction = model.predict(input_data)
            predicted_status = "Alive" if prediction[0] == 1 else "Dead"

            # Render results
            return render(request, "result.html", {"predicted_status": predicted_status})

        except Exception as e:
            return JsonResponse({"error": str(e)})

    return render(request, "predict_form.html")
