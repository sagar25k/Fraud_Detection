import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid GUI-related warnings
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess_data(csv_path):
    """Load and preprocess the dataset."""
    data = pd.read_csv(csv_path)
    pd.options.display.max_columns = None
    
    # Drop 'Time' column
    data = data.drop(['Time'], axis=1)
    
    # Scale 'Amount' feature
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(pd.DataFrame(data['Amount']))
    
    # Remove duplicate rows
    data = data.drop_duplicates()
    
    return data

def visualize_data(data, filename):
    """Visualize class distribution and save the plot to a file."""
    plt.style.use('ggplot')
    sns.countplot(x='Class', data=data)  # Use 'x' for specifying the axis
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Ensure the media directory exists
    vis_file_path = os.path.join(settings.MEDIA_ROOT, 'images', filename)
    os.makedirs(os.path.dirname(vis_file_path), exist_ok=True)
    plt.savefig(vis_file_path)
    plt.close()
    
    return vis_file_path

def train_model(X_train, y_train):
    """Train a Decision Tree Classifier model."""
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    return dtc

def evaluate_model(classifier, X_test, y_test):
    """Evaluate the model using various metrics."""
    y_pred = classifier.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

def perform_undersampling(data):
    """Perform undersampling to balance the dataset."""
    normal = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    normal_sample = normal.sample(n=fraud.shape[0], random_state=42)
    new_data = pd.concat([normal_sample, fraud], ignore_index=True)
    return new_data

def perform_oversampling(X, y):
    """Perform oversampling using SMOTE."""
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def save_model(model, pkl_path):
    """Save the trained model to a pickle file."""
    joblib.dump(model, pkl_path)

def load_model(pkl_path):
    """Load a model from a pickle file."""
    return joblib.load(pkl_path)

def make_prediction(model, input_data):
    """Make a prediction using the trained model."""
    pred = model.predict([input_data])
    return "Fraud Transaction" if pred[0] == 1 else "Normal Transaction"
