import numpy as np
import pandas as pd
import joblib
import seaborn as sns


from sklearn.preprocessing import LabelEncoder, scale, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

df=pd.read_csv('loan-train.xls')

df=df.drop('Loan_ID',axis=1)

df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())
df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

le=LabelEncoder()
df.Gender=le.fit_transform(df.Gender)
df.Married=le.fit_transform(df.Married)
df.Education=le.fit_transform(df.Education)
df.Self_Employed=le.fit_transform(df.Self_Employed)
df.Property_Area=le.fit_transform(df.Property_Area)
df.Loan_Status=le.fit_transform(df.Loan_Status)
df.Dependents=le.fit_transform(df.Dependents)

x=df.iloc[:,:-1]
y=df.Loan_Status

x_scale=pd.DataFrame(scale(x),columns=x.columns)
x_scale.head()

rus=RandomUnderSampler(sampling_strategy=1)
x_res,y_res=rus.fit_resample(x,y)
ax=y_res.value_counts().plot.pie(autopct='%.2f')
_=ax.set_title("under-sampling")

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=10)

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(x_res, y_res)

import joblib

# Save the trained model to a file
joblib.dump(clf, 'trained_model.joblib')



