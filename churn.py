import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
class BankChurn:
    def __init__(self):
        df = pd.read_csv('Churn_Modelling.csv')

        df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

        df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

        # '''

        # 	CreditScore	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Geography_Germany	Geography_Spain	Gender_Mal
        # 	619	         42	  2	    0.00	1	                 1  	   1	            101348.88	    	     0	            0	          0
        # '''






        x = df.drop(columns=['Exited'])
        y = df['Exited'].values
        print(x.columns)
        # print(df)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        scaled = StandardScaler()

        x_train_scaled = scaled.fit_transform(x_train)
        x_test_scaled = scaled.transform(x_test)
        # LOGISTIC REGRESSIONS
        # from sklearn.tree import DecisionTreeRegressor
        # clf = DecisionTreeRegressor(random_state=0)
        # fitting_data= clf.fit(x_train,y_train)
        # y_prediction = clf.predict(x_test)


        # accuracy= accuracy_score(y_test,y_prediction)
        # print(accuracy)
        # f1 =  f1_score(y_test,y_prediction)
        # prec =  precision_score(y_test,y_prediction)
        # rec = recall_score(y_test,y_prediction)
        # results=pd.DataFrame([['Logistic regression',accuracy,f1,prec,rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
        # # print(results)



        #RANDOM FOREST CLASSIFIER
        self.rfc = RandomForestClassifier(random_state=0)
        fitting_data_tree= self.rfc.fit(x_train,y_train)
        rfc_y_prediction = self.rfc.predict(x_test)

        ref_accuracy= accuracy_score(y_test,rfc_y_prediction)
        ref_f1 =  f1_score(y_test,rfc_y_prediction)
        rfc_prec =  precision_score(y_test,rfc_y_prediction)
        rfc_rec = recall_score(y_test,rfc_y_prediction)
        rfc_results=pd.DataFrame([['Random Forest Classifier',ref_accuracy,ref_f1, rfc_prec,rfc_rec]],columns=['Model','Accuracy','F1','Precision','Recall'])
        # print(ref_accuracy)


        # single_prediction=[[619,	42,	2,	0.0,	1,	1,	1,	101348.88,	0,	0,	0]]
        # # single_prediction=[[608	,41,	1	,83807.86,	1	,0	,1	,112542.58,	0,	1,	0]]
        # single_prediction=[[619,42,3, 23291.00,	3,	1,	1,	101338.88,	1,	1,	0]]
    def New_prediction(self,new_prediction):
        # print("rfc ",self.rfc.predict([new_prediction])[0])
        return self.rfc.predict([new_prediction])[0]
    
# obj = BankChurn()
# objec= obj.New_prediction([619,	42,	2,	0.0,	1,	1,	1,	101348.88,	0,	0,	0])
# print(objec)