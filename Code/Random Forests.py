import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


# Loads the dataset from the spreadsheet file
data_file_path = 'training data.csv'
data = pd.read_csv(data_file_path)

# Cleans the data by removing the ID column
columns_to_drop = ['id']
data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Splitting the dataset into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1875, random_state=42)

# Initializing and training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred_test = rf_model.predict(X_test)
y_pred_val = rf_model.predict(X_val)

# Begins calculating the metrics we're measuring
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])


# Outputting the evaluation metrics
print(f"Test Set Metrics:\nAccuracy: {test_accuracy:.5f}, Precision: {test_precision:.5f}, Recall: {test_recall:.5f}, F1-Score: {test_f1:.5f}, AUC: {test_auc:.5f}")


# Saving the model and the label encoder
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')


