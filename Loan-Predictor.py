import numpy as np
import pandas as pd
import matplotlib as pt
from matplotlib.pylab import plot, show
##plot().figure(10)
df = pd.read_csv(r"C:\Users\RAHUL\Desktop\Machine Learning\ProjectML-Loans\train.csv")  ##Reading data set

#############################################################################################################3
#print(df.head(10))
##df.describe()
#df['ApplicantIncome'].hist(bins=50)
#df.boxplot(column='ApplicantIncome')
#show()
#df['LoanAmount'].hist(bins=50)
#df.boxplot(column='LoanAmount')
#show()
#################################################################################################################3


##probability of getting a loan on basis of  cfedit history
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

####################################################################################################################
#print ('Frequency Table for Credit History:') 
#print (temp1)
#print ('\nProbility of getting loan for each Credit History class:')
#print (temp2)
#################################################################################################################

################################################Data Munging##############################################################
#### Data munging will be required in since multiple values are non-existant there. Different equations can be used 
#### to fill those non-existant data. like we can use mean or other calculated values.
sum1 = df.apply(lambda x: sum(x.isnull()),axis=0)				# see how many are null
#print(sum1)

###############simpler way to fill loanamount######################################33
#df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
#print(df.head(10))

##################################################################3
#On the basis of education and employed 
temp3 = df['Self_Employed'].value_counts(ascending=True)
#print(temp3)
# since most of the people are not self employed. So its ok to introduce 'NO' on missing values.
df['Self_Employed'].fillna('No',inplace=True)   
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
#print(table)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


df['Gender'].fillna('NaN',inplace=True)
df['Dependents'].fillna(0, inplace=True)
#table1 = df.pivot_table(values='Dependents', index='Married', aggfunc=np.mean)
#print(table1)

table1 = df.pivot_table(values='Married', index='Dependents', aggfunc=lambda x: x.map({'Yes':1,'No':0}).mean())
#print(table1)
df['Married'].fillna('No', inplace=True)
sum1 = df.apply(lambda x: sum(x.isnull()),axis=0)
#print(sum1)

table2 = df.pivot_table(values='Loan_Status', index='Credit_History', aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
#print(table2)

df['Credit_History'].fillna(0, inplace=True)
#print(df[df['Credit_History'].isnull() & df['Loan_Status'].iseq('Y'])#.fillna(0, inplace=True)
print(df.head(22))
###############################################################################
####################predictive model###########################################

############33 converting all values to numerical values###################
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
#for i in var_mod:
#	df[i], df.dtype = le.fit_transform(df[i])
print(le.fit_transform(df['Gender']))

#################################################################################3
################################################################################
#############sklearn############################################

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

def classification_model(model, data, predictors, outcome):
  #Fit the model:
	model.fit(data[predictors],data[outcome])
  
#Make predictions on training set:
	predictions = model.predict(data[predictors])

#Print accuracy
	accuracy = metrics.accuracy_score(predictions,data[outcome])
	print ("Accuracy : %s" % "{0:.3%}".format(accuracy))
	
	#Perform k-fold cross-validation with 5 folds
	kf = KFold(data.shape[0], n_folds=5)
	error = []
	for train, test in kf:
		# Filter training data										### divided here for checking how good the model is
		train_predictors = (data[predictors].iloc[train,:])
    
		# The target we're using to train the algorithm.
		train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
		model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
		error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))  # testing for all 
 
	print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
	model.fit(data[predictors],data[outcome]) 

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)

If the model ovrfits in any case, we can either chane the variables 
#featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
#print featimp
It will give theimportanceof features then feature selcetion can bedone on basisof intution