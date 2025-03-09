import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv('weather_dataset.csv')
# Features and target variable
features = [
    'Maximum Temperature', 'Minimum Temperature',
    'Maximum Apparent Temperature', 'Minimum Apparent Temperature', 
    'Precipitation Sum', 'Rain Sum', 'Showers Sum', 'Snowfall Sum', 
    'Precipitation Hours', 'Maximum Wind Speed', 'Maximum Wind Gusts', 
    'UV Index', 'UV Index Clear Sky'
]
target = 'Outfit'

X = data[features]
y = data[target]

# Preprocessing pipeline
numeric_features = [
    'Maximum Temperature', 'Minimum Temperature', 'Maximum Apparent Temperature', 
    'Minimum Apparent Temperature', 'Precipitation Sum', 'Rain Sum', 
    'Showers Sum', 'Snowfall Sum', 'Precipitation Hours', 'Maximum Wind Speed', 
    'Maximum Wind Gusts', 'UV Index', 'UV Index Clear Sky'
]


numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'outfit_recommender_model.joblib')

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')