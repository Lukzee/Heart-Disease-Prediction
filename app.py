from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64

# Import necessary classifiers from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset (ensure heart.csv is in the same directory)
data = pd.read_csv('heart.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets (for demonstration, training is done at startup)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
knn = KNeighborsClassifier().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC(probability=True).fit(X_train, y_train)
lr = LogisticRegression(max_iter=2000).fit(X_train, y_train)

# Home route that renders the input form
@app.route('/')
def index():
    # Pass the list of feature names (as in the CSV columns except target)
    feature_cols = X.columns.tolist()
    # Sample data based on common UCI Heart Disease dataset features
    known_samples = {
        "age": "63",
        "sex": "1 for male, 0 for female",
        "cp": "3 (typical angina)",
        "trestbps": "145",
        "chol": "233",
        "fbs": "1 if true, 0 if false",
        "restecg": "2 (abnormality detected)",
        "thalach": "150",
        "exang": "1 if yes, 0 if no",
        "oldpeak": "2.3",
        "slope": "3",
        "ca": "0",
        "thal": "3"
    }
    # Fallback placeholder for any unexpected feature names
    sample_data = {col: known_samples.get(col, "e.g., value") for col in feature_cols}
    return render_template('index.html', feature_cols=feature_cols, sample_data=sample_data)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve features from data
    features = []
    for col in X.columns:
        features.append(float(request.form.get(col)))
    input_data = np.array(features).reshape(1, -1)

    # Dictionary of models
    models = {
        'K-Nearest Neighbors': knn,
        'Decision Tree': dt,
        'Random Forest': rf,
        'Support Vector Machine': svm,
        'Logistic Regression': lr
    }

    results = {}
    # Get predictions and probabilities (if available) from each model
    for name, model in models.items():
        prediction = model.predict(input_data)[0]
        # Check if model has predict_proba, then convert the array to a list
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0].tolist()  # Convert numpy array to list
        else:
            prob = None
        results[name] = {'prediction': prediction, 'probability': prob}

    # Generate graphs for each model's prediction probabilities
    graphs = {}
    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        if results[name]['probability'] is not None:
            probs = results[name]['probability']
            ax.bar(['No Disease', 'Disease'], probs, color=['blue', 'red'])
            ax.set_ylim([0, 1])
            ax.set_title(f'{name} Probabilities')
        else:
            # In case predict_proba is not available, simply display the prediction
            ax.text(0.5, 0.5, f'Prediction: {results[name]["prediction"]}', horizontalalignment='center',
                    verticalalignment='center')
            ax.axis('off')
        # Convert plot to PNG image and then to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        graphs[name] = image_base64
        plt.close(fig)

    # The ultimate result will be the Random Forest prediction
    final_result = results['Random Forest']['prediction']

    return render_template('result.html', results=results, graphs=graphs, final_result=final_result)

if __name__ == '__main__':
    app.run(debug=True)