# app.py
import pickle
from flask import Flask, request, render_template
from model import PreprocessingData

flask_app = Flask(__name__)

dataset_path = "dataset/Stroke_dataset.csv"
dataM = PreprocessingData()
dataM.proses(dataset_path)
dataM.DataSelection()

def convert_smoking_status_to_numeric(smoking_status):
    if smoking_status == 'never_smoked':
        return 0
    elif smoking_status == 'formerly_smoked':
        return 1
    elif smoking_status == 'smokes':
        return 2
    elif smoking_status == 'unknown':
        return 3
    else:
        return -1

@flask_app.route("/", methods=["POST", "GET"], endpoint='home')
def home():
    result = ''
    dataInputan = {}

    if request.method == 'POST':
        age = int(request.form["age"])
        gender = request.form["gender"]
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        avg_glucose_level = float(request.form["avg_glucose_level"])
        bmi = float(request.form["bmi"])
        smoking_status = request.form["smoking_status"]
        work_type = request.form["work_type"]
        metode = request.form["metode"]

        # Convert smoking_status to numeric using the previous function
        smoking_status_numeric = convert_smoking_status_to_numeric(smoking_status)

        # Convert work_type and gender to numeric using LabelEncoders
        work_type_numeric = dataM.label_encoders['work_type'].transform([work_type])[0]
        gender_numeric = dataM.label_encoders['gender'].transform([gender])[0]

        # Ensure consistent number of features with the trained model
        selected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi'] + \
                           [col for col in dataM.dataset.columns if 'smoking_status' in col]
        input_features = [
            [gender_numeric, age, hypertension, heart_disease, work_type_numeric, avg_glucose_level, bmi, smoking_status_numeric, 0, 0]
        ]

        if metode == 'knn':
            dataM.MetodeKnn()
            model = pickle.load(open("model/stroke_KNN.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = {'age': age, 'gender': gender, 'hypertension': hypertension, 'heart_disease': heart_disease,
                           'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'smoking_status': smoking_status,
                           'work_type': work_type, 'metode': metode}
            return render_template("index.html", result=result, dataInputan=dataInputan)

        elif metode == 'naive_bayes':
            dataM.MetodeNaiveBayes()
            model = pickle.load(open("model/Stroke_NB.pkl", "rb"))
            result = model.predict(input_features)[0]
            dataInputan = {'age': age, 'gender': gender, 'hypertension': hypertension, 'heart_disease': heart_disease,
                           'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'smoking_status': smoking_status,
                           'work_type': work_type, 'metode': metode}
            return render_template("index.html", result=result, dataInputan=dataInputan)

    # Clear result and dataInputan when not a POST request
    result = ''
    dataInputan = {}
    return render_template("index.html", result=result, dataInputan=dataInputan)

if __name__ == "__main__":
    flask_app.run(debug=True)
