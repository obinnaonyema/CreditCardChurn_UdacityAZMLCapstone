from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

cc_data = "https://raw.githubusercontent.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/main/BankChurners.csv"
cc_cust = TabularDatasetFactory.from_delimited_files(path=cc_data, separator=',')
columns_not_needed = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2','CLIENTNUM']
cc_final = cc_cust.drop_columns(columns_not_needed)

# Save model for current iteration

run = Run.get_context()

x_df = cc_final.to_pandas_dataframe()

#convert to numeric
x_df.Attrition_Flag.replace({'Existing Customer': 1, 'Attrited Customer': 0}, inplace=True)

x_df.Gender.replace({'F': 0, 'M': 1}, inplace=True)

x_df.Education_Level.replace({'Graduate': 0, 'High School': 1, 'Unknown': 2,'Uneducated':3,'College':4,'Post-Graduate':5,'Doctorate':6}, inplace=True)

x_df.Marital_Status.replace({'Married': 0, 'Single': 1,'Unknown':2,'Divorced':3}, inplace=True)

x_df.Income_Category.replace({'Less than $40K': 0,'$80K - $120K': 1, '$60K - $80K': 2,'Unknown':3}, inplace=True)
x_df.Income_Category.replace({'$40K - $60K': 4}, inplace=True)
x_df.Income_Category.replace({'$120K +': 5}, inplace=True)

x_df.Card_Category.replace({'Blue': 0, 'Silver': 1,'Gold':2,'Platinum':3}, inplace=True)

#define the label
y_df = x_df.pop("Attrition_Flag")

# TODO: Split data into train and test sets.

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=1)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    #Save model for current iteration, also include the value for C and max_iter in filename, random_state=
    os.makedirs('outputs', exist_ok=True)

    joblib.dump(model, filename='./outputs/cchypermodel.pkl')

if __name__ == '__main__':
    main()