import pandas as pd
import joblib
import sys

def choose_model():
    while True:
        print("Choose a machine learning model for prediction:")
        print("1. Gradient Boosting Machine")
        print("2. SVM")
        print("3. Random Forest")
        print("4. Logistic Regression")


        choice = input("Enter the number of your choice: ").strip()

        if choice == '1':
            return 'gbm_model.pkl'
        elif choice == '2':
            return 'svm_model.pkl'
        elif choice == '3':
            return 'rf_model.pkl'
        elif choice == '4':
            return 'lr_model.pkl'

        else:
            print("Invalid choice. Please enter a valid number.")
            # The loop continues if the input is invalid


# Function to collect user input for new patient data
def get_patient_data():
    descriptions = {
        "radius_mean": "Average size of the tumor radius (in cm): ",
        "texture_mean": "Average texture of the tumor : ",
        "perimeter_mean": "Average perimeter of the tumor (in cm): ",
        "area_mean": "Average area of the tumor (in cm²): ",
        "smoothness_mean": "Average smoothness of the tumor surface : ",
        "compactness_mean": "Average compactness of the tumor : ",
        "concavity_mean": "Average number of concave portions of the tumor : ",
        "concave points_mean": "Average number of concave points on the tumor : ",
        "symmetry_mean": "Average symmetry of the tumor : ",
        "fractal_dimension_mean": "Average 'fractal dimension' of the tumor: ",
        "radius_se": "Change in tumor radius size from average: ",
        "texture_se": "Change in tumor texture from average: ",
        "perimeter_se": "Change in tumor perimeter from average: ",
        "area_se": "Change in tumor area from average: ",
        "smoothness_se": "Change in tumor surface smoothness from average: ",
        "compactness_se": "Change in tumor compactness (density) from average: ",
        "concavity_se": "Change in number of concave portions of tumor from average: ",
        "concave points_se": "Change in number of concave points on tumor from average: ",
        "symmetry_se": "Change in tumor symmetry from average: ",
        "fractal_dimension_se": "Change in tumor 'fractal dimension' (complexity) from average: ",
        "radius_worst": "Largest size of the tumor radius recorded (in cm): ",
        "texture_worst": "Roughest texture of the tumor recorded: ",
        "perimeter_worst": "Largest perimeter of the tumor recorded (in cm): ",
        "area_worst": "Largest area of the tumor recorded (in cm²): ",
        "smoothness_worst": "Roughest surface of the tumor recorded: ",
        "compactness_worst": "Highest compactness of the tumor recorded: ",
        "concavity_worst": "Maximum number of concave portions of tumor recorded: ",
        "concave points_worst": "Maximum number of concave points on tumor recorded: ",
        "symmetry_worst": "Highest symmetry of the tumor recorded: ",
        "fractal_dimension_worst": "Highest 'fractal dimension' of the tumor recorded : "
    }

    patient_data = {}
    if len(sys.argv) > 1:  # Data passed as command-line arguments
        for i, feature in enumerate(descriptions.keys(), start=1):
            patient_data[feature] = float(sys.argv[i])
    else:  # Original manual data entry
        for feature in descriptions.keys():
            patient_data[feature] = float(input(descriptions[feature]))

    return patient_data

# Function to predict cancer

def predict_cancer(model_file, patient_details):
    # Load the chosen model
    model = joblib.load(model_file)

    # Load the label encoder
    encoder = joblib.load('label_encoder.pkl')

    # Convert the patient details into a DataFrame
    patient_df = pd.DataFrame([patient_details], columns=patient_details.keys())

    # Make a prediction
    prediction = model.predict(patient_df)

    # Use the label encoder to get the categorical label
    prediction_label = encoder.inverse_transform(prediction)

    # Return "Cancerous" if the prediction label is 'M', otherwise "Non-cancerous"
    return "CANCEROUS" if prediction_label[0] == 'M' else "NON-CANCEROUS"


def main():
    model_file = choose_model()
    while True:
        new_patient_data = get_patient_data()
        prediction_result = predict_cancer(model_file, new_patient_data)
        print(f"The prediction for the entered patient data is: {prediction_result}")

        user_choice = input("Do you want to enter data for another patient? (yes/no): ").strip().lower()
        if user_choice != 'yes':
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()

