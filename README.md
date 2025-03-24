
# COVID Symptom Analysis

COVID Symptom Analysis is a web application that allows users to input symptoms and receive a prediction of whether the symptoms could be related to COVID-19. The application uses a machine learning model to make predictions based on user input and provides a quick way to analyze symptoms for potential COVID cases.

## Features

- Predicts whether a person is likely to have COVID based on symptoms.
- Uses a pre-trained Random Forest model (`model_rf.joblib`).
- User-friendly web interface built with Flask.
- Easy to extend with additional features.

## Technologies Used

- **Flask**: A lightweight web framework for Python.
- **Joblib**: For loading the pre-trained machine learning model.
- **Numpy**: For handling numerical operations.
- **Random Forest Classifier**: The machine learning model used for prediction.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/covid_symptom_analysis.git
cd covid_symptom_analysis
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

The application will run locally at `http://127.0.0.1:5000/`.

## Model Information

The machine learning model (`model_rf.joblib`) is a Random Forest Classifier trained on a dataset of COVID-19 symptoms. The model predicts the likelihood of a person having COVID based on the following symptoms:

- Fever
- Cough
- Fatigue
- Shortness of breath
- Loss of taste or smell
- And more...

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
