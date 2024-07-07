# file for training models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from preprocess import *

students_performance = pd.read_csv('students_performance.csv')
data = data_processing(students_performance)
X_train, X_test, y_train, y_test = data_split(data)

X_train_scaled, X_test_scaled = scaling_data(X_train, X_test)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=9),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred)
    print(f'{model_name} F1 Score: {f1:.2f}')


