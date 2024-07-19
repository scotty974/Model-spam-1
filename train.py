import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, validation_curve
from tqdm import tqdm
import engine
from sklearn.model_selection import ParameterGrid
import pickle

# Load the data
data = pd.read_csv('spam.csv', encoding="ISO-8859-1", sep=',')

# Clean the data
data_clean = data[['v1', 'v2']]

# Encode labels
lb = LabelEncoder()
data_clean['v1'] = lb.fit_transform(data_clean['v1'])

X = data_clean['v2']
y = data_clean.v1

# Vectorize text data
tfid = TfidfVectorizer()
X = tfid.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNeighborsClassifier
model = KNeighborsClassifier()

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
print('Entrainement : ', model.score(X_train, y_train))
print('Test : ', model.score(X_test, y_test))

# Cross-validation with tqdm progress bar
print("Cross validation...")
scores = cross_val_score(model, X, y, cv=5)
print('Cross validation : ', scores.mean())

# Validation curve with tqdm progress bar
alpha_range = np.arange(1, 50)
print("Validation curve...")
train_scores = []
val_scores = []
for alpha in tqdm(alpha_range):
    train_score, val_score = validation_curve(
        model, X_train, y_train,
        param_name="n_neighbors",
        param_range=[alpha],
        cv=5
    )
    train_scores.append(train_score.mean())
    val_scores.append(val_score.mean())

print('Entrainement : ', np.mean(train_scores))
print('Test : ', np.mean(val_scores))

# Grid search with tqdm progress bar
paramGrid = {
    'n_neighbors': np.arange(1, 20),
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

param_list = list(ParameterGrid(paramGrid))
print("Grid search...")

best_score = 0
best_params = None
for params in tqdm(param_list):
    grid = KNeighborsClassifier(**params)
    grid.fit(X_train, y_train)
    score = cross_val_score(grid, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_params = params
        best_model = grid

print('Best score : ', best_score)
print('Best parameters : ', best_params)

with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfid, f)