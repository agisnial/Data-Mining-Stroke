import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

class PreprocessingData():
    def __init__(self, dataset=None, x_test=None, y_test=None, x_train=None, y_train=None):
        self.x_test = x_test
        self.y_test = y_test
        self.x_train = x_train
        self.y_train = y_train
        self.dataset = dataset
        self.label_encoders = {}

    def proses(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)

        # Ganti nilai yang hilang pada kolom numerik dengan mean
        numerical = [var for var in self.dataset.columns if self.dataset[var].dtype != 'O']
        imputer = SimpleImputer(strategy='mean')
        self.dataset[numerical] = imputer.fit_transform(self.dataset[numerical])

        # Check if 'smoking_status' column exists
        if 'smoking_status' not in self.dataset.columns:
            # If not, create it with a default value ('Unknown')
            self.dataset['smoking_status'] = 'Unknown'

        # Handle missing values in 'smoking_status' column
        self.dataset['smoking_status'].fillna('Unknown', inplace=True)

        # Perform one-hot encoding for 'smoking_status'
        self.dataset = pd.get_dummies(self.dataset, columns=['smoking_status'], prefix='smoking_status', drop_first=True)

        # Convert categorical variables to numeric using LabelEncoder
        categorical_vars = ['gender', 'ever_married', 'work_type', 'Residence_type']
        for var in categorical_vars:
            le = LabelEncoder()
            self.dataset[var] = le.fit_transform(self.dataset[var])
            self.label_encoders[var] = le

        print(self.dataset['stroke'].value_counts(normalize=True))
        return self.dataset

    def DataSelection(self, method='knn'):
        # Select relevant columns including one-hot encoded 'smoking_status'
        selected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi'] + \
                           [col for col in self.dataset.columns if 'smoking_status' in col]

        if method == 'knn':
            # For KNN, include all features (no need to drop one-hot encoded columns)
            selected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi'] + \
                               [col for col in self.dataset.columns if 'smoking_status' in col]

        x = self.dataset[selected_columns]
        y = self.dataset['stroke']

        # Split the dataset into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.025, random_state=0)

        # Print the resulting datasets
        print(self.x_train, self.x_test, self.y_train, self.y_test)

    def MetodeKnn(self):
        sc = StandardScaler()

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        self.x_train = imputer.fit_transform(self.x_train)
        self.x_test = imputer.transform(self.x_test)

        # Standardize the data
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        # Train KNN model with different n_neighbors
        best_k = self.find_best_k()
        knn_model = KNeighborsClassifier(n_neighbors=best_k)
        knn_model.fit(self.x_train, self.y_train)

        # Evaluasi model
        accuracy_knn = knn_model.score(self.x_test, self.y_test)
        print(f'Accuracy of KNN model: {accuracy_knn}')

        # Print confusion matrix and classification report
        y_pred_knn = knn_model.predict(self.x_test)
        print("Confusion Matrix (KNN):\n", confusion_matrix(self.y_test, y_pred_knn))
        print("Classification Report (KNN):\n", classification_report(self.y_test, y_pred_knn))

        # Create the 'model/' directory if it doesn't exist
        os.makedirs('model', exist_ok=True)

        # Save the model
        pickle.dump(knn_model, open("model/stroke_KNN.pkl", "wb"))

    def find_best_k(self):
        # Function to find the best value for n_neighbors
        k_values = [3, 5, 7, 9, 11]  # You can expand this list
        best_accuracy = 0
        best_k = 0

        for k in k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)
            knn_model.fit(self.x_train, self.y_train)
            accuracy = knn_model.score(self.x_test, self.y_test)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        print(f'Best k for KNN: {best_k} with accuracy: {best_accuracy}')
        return best_k

    def MetodeNaiveBayes(self):
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        self.x_train = imputer.fit_transform(self.x_train)
        self.x_test = imputer.transform(self.x_test)

        # Train Naive Bayes model
        nb_model = GaussianNB()
        nb_model.fit(self.x_train, self.y_train)

        # Evaluasi model
        accuracy_nb = nb_model.score(self.x_test, self.y_test)
        print(f'Accuracy of Naive Bayes model: {accuracy_nb}')

        # Print confusion matrix and classification report
        y_pred_nb = nb_model.predict(self.x_test)
        print("Confusion Matrix (Naive Bayes):\n", confusion_matrix(self.y_test, y_pred_nb))
        print("Classification Report (Naive Bayes):\n", classification_report(self.y_test, y_pred_nb))

        # Save the model
        pickle.dump(nb_model, open("model/Stroke_NB.pkl", "wb"))

if __name__ == "__main__":
    dataM = PreprocessingData()
    dataM.proses("dataset/Stroke_dataset.csv")
    dataM.DataSelection(method='knn')
    dataM.MetodeKnn()
    dataM.MetodeNaiveBayes()
