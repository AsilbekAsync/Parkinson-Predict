from brain import predict

fields = {
    "Age": 85,
    "Gender": 0,
    "Ethnicity": 3,
    "EducationLevel": 1,
    "BMI": 19.6,
    "Smoking": 0,
    "AlcoholConsumption": 5.1,
    "PhysicalActivity": 1.38,
    "DietQuality": 3.89,
    "SleepQuality": 9.28,
    "FamilyHistoryParkinsons": 0,
    "TraumaticBrainInjury": 0,
    "Hypertension": 0,
    "Diabetes": 0,
    "Depression": 0,
    "Stroke": 0,
    "SystolicBP": 129,
    "DiastolicBP": 60,
    "CholesterolTotal": 222.8,
    "CholesterolLDL": 148.1,
    "CholesterolHDL": 37.8,
    "CholesterolTriglycerides": 337.3,
    "UPDRS": 6.45,
    "MoCA": 29.18,
    "FunctionalAssessment": 1.57,
    "Tremor": 1,
    "Rigidity": 0,
    "Bradykinesia": 0,
    "PosturalInstability": 0,
    "SpeechProblems": 0,
    "SleepDisorders": 0,
    "Constipation": 0
}

new_data = {}

for key, default_value in fields.items():
    while True:
        user_input = input(f"{key} ({default_value}): ").strip()
        if not user_input:
            new_data[key] = default_value
            break
        else:
            try:
                if isinstance(default_value, int):
                    val = int(user_input)
                else:
                    val = float(user_input)
                new_data[key] = val
                break
            except ValueError:
                print("Iltimos, to'g'ri raqam kiriting!")


prediction = predict(new_data, "models/model.joblib")
print("Modelning bashorati:", "Parkinsons kasalligi aniqlandi" if prediction == 1 else "Sog'lom")