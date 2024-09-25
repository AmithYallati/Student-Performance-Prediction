import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('E:/last/xAPI-Edu-Data.csv')  # Updated file path

# Step 1: Data Preprocessing
# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic',
                       'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
                       'StudentAbsenceDays', 'Class']

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features (X) and target variable (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical columns
scaler = StandardScaler()
numerical_columns = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Step 2: Train the RandomForestClassifier model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 3: Make predictions and evaluate the model
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", report)
