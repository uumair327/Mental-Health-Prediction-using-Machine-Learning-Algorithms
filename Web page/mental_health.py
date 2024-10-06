import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("C:/Users/uumai/OneDrive/Desktop/Mental-Health-Prediction-using-Machine-Learning-Algorithms/Web page/mental_health.csv")

# Define lists for gender mapping
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr", "cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", 
             "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female_str = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail"]

# Standardize the Gender column
for (row, col) in data.iterrows():
    gender_lower = str.lower(col.Gender)
    if gender_lower in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)
    elif gender_lower in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)
    elif gender_lower in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

# Get rid of unwanted values
stk_list = ['A little about you', 'p']
data = data[~data['Gender'].isin(stk_list)]

# Map categorical values to numeric
data['Gender'] = data['Gender'].map({'male': 0, 'female': 1, 'trans': 2})
data['family_history'] = data['family_history'].map({'No': 0, 'Yes': 1})
data['treatment'] = data['treatment'].map({'No': 0, 'Yes': 1})

# Prepare the features (X) and target (y) variables
# Ensure you are selecting the correct columns: age, gender, and family history
X = data[['Age', 'Gender', 'family_history']].values  # Only select the relevant 3 columns
y = data['treatment'].values  # Target column

# Convert the arrays to integer type
X = X.astype(int)
y = y.astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize classifiers
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

# Stack classifiers using a meta-classifier (Logistic Regression)
stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

# Train the stacked model
stack.fit(X_train, y_train)

# Save the trained model to a file
pickle.dump(stack, open('Web page/model.pkl', 'wb'))

# Load the model from the file for future use
model = pickle.load(open('Web page/model.pkl', 'rb'))
