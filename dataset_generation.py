import pandas as pd
import numpy as np
import random

def generate_covid_dataset(n=20000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Generate demographics
    age = np.random.randint(18, 90, size=n)
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.45, 0.55])
    diabetes = np.random.choice([0, 1], size=n, p=[0.8, 0.2])  # 20% have diabetes
    hypertension = np.random.choice([0, 1], size=n, p=[0.8, 0.2])  # 20% have hypertension
    obesity = np.random.choice([0, 1], size=n, p=[0.8, 0.2])  # 20% have obesity

    # Female-specific attribute
    pregnant = [1 if gender[i] == "Female" and 18 <= age[i] <= 45 and random.random() < 0.08 else 0 for i in range(n)]  # 8% of females are pregnant

    # Generate symptoms with realistic probabilities
    fever = np.random.choice([0, 1], size=n, p=[0.4, 0.6])  # 60% chance of fever
    cough = np.random.choice([0, 1], size=n, p=[0.35, 0.65])  # 65% chance of cough
    fatigue = np.random.choice([0, 1], size=n, p=[0.5, 0.5])  # 50% chance of fatigue
    shortness_of_breath = np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 30% have shortness of breath
    loss_of_taste_smell = np.random.choice([0, 1], size=n, p=[0.6, 0.4])  # 40% have loss of taste/smell
    headache = np.random.choice([0, 1], size=n, p=[0.5, 0.5])  # 50% chance of headache
    sore_throat = np.random.choice([0, 1], size=n, p=[0.55, 0.45])  # 45% chance of sore throat
    body_aches = np.random.choice([0, 1], size=n, p=[0.5, 0.5])  # 50% chance of body aches

    # Count total symptoms per patient
    symptom_counts = (
        fever + cough + fatigue + shortness_of_breath + 
        loss_of_taste_smell + headache + sore_throat + body_aches
    )

    # Severity classification based on symptoms and comorbidities
    severity = []
    for i in range(n):
        if symptom_counts[i] >= 6 or (shortness_of_breath[i] == 1 and (diabetes[i] == 1 or hypertension[i] == 1 or obesity[i] == 1)):
            severity.append("Severe")
        elif 4 <= symptom_counts[i] < 6:
            severity.append("Moderate")
        elif 1 <= symptom_counts[i] < 4:
            severity.append("Mild")
        else:
            severity.append("Asymptomatic")

    # Realistic Test Result Logic
    test_result = []
    for i in range(n):
        has_major_symptoms = shortness_of_breath[i] == 1 or loss_of_taste_smell[i] == 1 or fever[i] == 1
        has_mild_symptoms = cough[i] == 1 or headache[i] == 1 or body_aches[i] == 1 or sore_throat[i] == 1

        # High-risk group (more likely to test positive)
        if has_major_symptoms and symptom_counts[i] >= 4:
            test_result.append("Positive" if random.random() < 0.75 else "Negative")
        elif has_major_symptoms and symptom_counts[i] >= 2:
            test_result.append("Positive" if random.random() < 0.6 else "Negative")
        elif has_mild_symptoms and symptom_counts[i] >= 4:
            test_result.append("Positive" if random.random() < 0.3 else "Negative")  # Lower probability for mild symptoms
        elif has_mild_symptoms and symptom_counts[i] < 4:
            test_result.append("Negative")  # Most mild-symptom cases should be negative
        else:
            test_result.append("Negative")  # Asymptomatic cases mostly negative

    # Create DataFrame
    df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "Pregnant": pregnant,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Obesity": obesity,
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue,
        "Shortness_of_Breath": shortness_of_breath,
        "Loss_of_Taste_Smell": loss_of_taste_smell,
        "Headache": headache,
        "Sore_Throat": sore_throat,
        "Body_Aches": body_aches,
        "Total_Symptoms": symptom_counts,
        "Severity": severity,
        "Test_Result": test_result
    })

    return df

# Generate dataset and save to CSV
df = generate_covid_dataset(n=20000)
df.to_csv("covid_symptom_dataset.csv", index=False)


