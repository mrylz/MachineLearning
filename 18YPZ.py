import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,LabelEncoder,RobustScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,classification_report
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("14-income_evaluation.csv")
col_names = ['age', 'workclass', 'finalweight', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df.columns = col_names
categorical = [col for col in df.columns if df[col].dtype=='O']
numerical = [col for col in df.columns if df[col].dtype!='O']
df['workclass']=df['workclass'].replace(' ?', np.nan)
df['occupation']=df['occupation'].replace(' ?', np.nan)
df['native_country']=df['native_country'].replace(' ?', np.nan)
X = df.drop(['income'], axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
for i in [X_train, X_test]:
    i['workclass'] = i['workclass'].fillna(X_train['workclass'].mode()[0])
    i['occupation'] = i['occupation'].fillna(X_train['occupation'].mode()[0])
    i['native_country'] = i['native_country'].fillna(X_train['native_country'].mode()[0])
y_train_binary = y_train.apply(lambda x: 1 if x.strip() == '>50K' else 0)
target_means = y_train_binary.groupby(X_train['native_country']).mean()
X_train['native_country'] = X_train['native_country'].map(target_means)
X_train['native_country'] = X_train['native_country'].fillna(y_train_binary.mean())
X_test['native_country'] = X_test['native_country'].map(target_means)
X_test['native_country'] = X_test['native_country'].fillna(y_train_binary.mean())
le = LabelEncoder()
X_train['workclass']=X_train['workclass'].str.strip()
X_test['workclass']=X_test['workclass'].str.strip()
X_train['workclass'] = le.fit_transform(X_train['workclass'])
X_test['workclass'] = le.transform(X_test['workclass'])
X_train['education']=X_train['education'].str.strip()
X_test['education']=X_test['education'].str.strip()
X_train['education'] = le.fit_transform(X_train['education'])
X_test['education'] = le.transform(X_test['education'])
X_train['marital_status']=X_train['marital_status'].str.strip()
X_test['marital_status']=X_test['marital_status'].str.strip()
X_train['marital_status'] = le.fit_transform(X_train['marital_status'])
X_test['marital_status'] = le.transform(X_test['marital_status'])
X_train['occupation']=X_train['occupation'].str.strip()
X_test['occupation']=X_test['occupation'].str.strip()
X_train['occupation'] = le.fit_transform(X_train['occupation'])
X_test['occupation'] = le.transform(X_test['occupation'])
X_train['relationship']=X_train['relationship'].str.strip()
X_test['relationship']=X_test['relationship'].str.strip()
X_train['relationship'] = le.fit_transform(X_train['relationship'])
X_test['relationship'] = le.transform(X_test['relationship'])
X_train['race']=X_train['race'].str.strip()
X_test['race']=X_test['race'].str.strip()
X_train['race'] = le.fit_transform(X_train['race'])
X_test['race'] = le.transform(X_test['race'])
X_train['sex']=X_train['sex'].str.strip()
X_test['sex']=X_test['sex'].str.strip()
X_train['sex'] = le.fit_transform(X_train['sex'])
X_test['sex'] = le.transform(X_test['sex'])
y_train = y_train.str.strip()
y_test = y_test.str.strip()
y_train= le.fit_transform(y_train)
y_test= le.transform(y_test)
scaled = RobustScaler()
X_train_scaled = scaled.fit_transform(X_train)
X_test_scaled = scaled.transform(X_test)
reg = RandomForestClassifier(n_estimators=100, random_state=15)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f'Model accuracy score with default decision-trees : {accuracy_score(y_test, y_pred)}')

