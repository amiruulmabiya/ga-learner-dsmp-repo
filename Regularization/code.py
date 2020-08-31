# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path

#Code starts here

# load the dataframe
df = pd.read_csv(path)

#Indepenent varibles
X = df.drop('Price',axis=1)

# store dependent variable
y = df['Price']

# spliting the dataframe

X_train,X_test,y_train,y_test=train_test_split(X,y ,test_size=0.3,random_state=6)

# check correlation
corr=X_train.corr()

# print correlation
print(corr)

#Code ends here




# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred =regressor.predict(X_test)
r2 = regressor.score (X_test,y_test)
print(r2)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso =Lasso()
lasso.fit(X_train ,y_train)
lasso_pred =lasso.predict(X_test)
r2_lasso = lasso.score (X_test,y_test)


# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge =Ridge()
ridge.fit(X_train ,y_train)
ridge_pred =ridge.predict(X_test)
r2_ridge = ridge.score (X_test,y_test)



# Code ends here


# --------------
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
score = cross_val_score(regressor,X_train,y_train,cv=10)
mean_score =np.mean(score)
print(mean_score)
#Code starts here


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train ,y_train)
y_pred= model.predict(X_test)
r2_poly = model.score (X_test,y_test)
print(r2_poly)
#Code starts here


