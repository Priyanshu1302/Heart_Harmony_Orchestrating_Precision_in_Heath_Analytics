import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sample_heart = pd.read_csv(r"heart_2022_with_nans.csv")
print(sample_heart)
sample_heart.shape
heart = sample_heart.sample(frac=0.05)
heart.shape
heart. drop_duplicates(inplace=True)
heart.duplicated().sum()
heart.isnull()
heart.isnull().any()
heart.dropna(how="all")
heart.dropna(subset=['GeneralHealth','PhysicalHealthDays','MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities','SleepHours', 'RemovedTeeth', 'HadHeartAttack', 'HadAngina',
       'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory','HeightInMeters','WeightInKilograms','BMI','AlcoholDrinkers','HIVTesting','FluVaxLast12',
                      'PneumoVaxEver','TetanusLast10Tdap','HighRiskLastYear','CovidPos'], inplace=True)
heart.isnull().sum()
from sklearn.preprocessing import LabelEncoder
columns_to_convert = ['State', 'Sex', 'GeneralHealth'
                      , 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth', 'HadHeartAttack', 'HadAngina',
       'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
       'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
       'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
       'DifficultyConcentrating', 'DifficultyWalking',
       'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus',
       'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory','AlcoholDrinkers','HIVTesting','FluVaxLast12',
                      'PneumoVaxEver','TetanusLast10Tdap','HighRiskLastYear','CovidPos']
label_encoder = LabelEncoder()
for col in columns_to_convert:
    if heart[col].dtype == 'object': heart[col] = label_encoder.fit_transform(heart[col])
heart["HadHeartAttack"].unique()
heart['HadHeartAttack'].value_counts(normalize=True)
X = heart.drop('HadHeartAttack', axis=1)
y = heart['HadHeartAttack']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.metrics import roc_curve, auc
X_train , X_test , y_train ,  y_test = train_test_split(X, y, train_size=0.7 , random_state=42)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f2_score = fbeta_score(y_test, y_pred, beta=2)

print("Precision:", precision)
print("Recall:", recall)
print("F2 Score:", f2_score)
rows_with_heart = heart[heart['HadHeartAttack'] == 1]
print(rows_with_heart)
import pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)
with open('model.pkl', 'rb') as model_file:
    logreg_from_pickle = pickle.load(model_file)