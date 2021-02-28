*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Credit Card Churn

This is my chosen project for the Capstone Project in the Azure Machine Learning Engineer Nanodegree by Udacity. In this project, I attempt to build a model that will help a bank manager predict customers that are likely to churn. The manager's intention is to proactively engage these customers with a view to preventing churn. 

2 methods are used: one with Automated ML and one with the hyperdrive service. The better model is selected and deployed and sample requests are sent to the deployed endpoint.

## Project Set Up and Installation

To run this experiment, try the following:
<ol>
  <li>Download this repository into your Azure ML Workspace. The raw data file is also provided in this repository.</li>
  <li>Set up your compute to run the jupyter notebook. You can use a DS2_V2 instance.</li>
  <li>Next you run each cell in the notebooks automl.ipynb, hyperparameter_tuning.ipynb. I ran each notebook one at a time to take note of the differences</li>
  <li>The notebooks contain the configuration to set up compute clusters for training the data.</li>
</ol>

This is the flow of the project as expected by the instructure:

![Project Flow](https://github.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/blob/main/Images/project_flow.PNG)

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.
The data set was obtained from Kaggle [here](https://www.kaggle.com/sakshigoyal7/credit-card-customers). It contains details of credit card customers of a bank. There are 22 columns and 10000 rows. The last 2 columns were advised to be discarded by the data set provider which I have done in this project.

The data contains details such as education status, number of dependents, inactivity period, transaction counts, credit limits and so on. Our label is Attrition_Flag which says whether a customer is an existing customer or attrited customer.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
The aim of the project is to predict the likelihood of churn, which is denoted by the records in the Attrition_Flag column of the raw data set. It is interesting to note that this dataset had some imbalance in the sense that only 1627 out of 10127 customers have churned.

### Access
*TODO*: Explain how you are accessing the data in your workspace.
I use TabularDatasetFactory class to download the data from my github repo. 

cc_cust = TabularDatasetFactory.from_delimited_files(path=cc_data, separator=',')

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment
I ran the experiment a few times using timeout settings of 20mins and 40mins but this was too short for any child runs to start. I set experiment_timeout_minutes to 60 so as to allow significant time for training of a few child models. Primary metric used is accuracy and I tried to set a metric exit score as after a few attempts at running the experiment I noticed the score doesn't improve much after the first 5 child runs.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
Although the AutoML experiment timeout setting ended the experiment, accuracy score was 0.95 for the best model. I ran the experiment more than one noticing that each time the accuracy score was different. Perhaps the difference in compute resources may have influenced the outcome. 

The image below shows run details:

![Run details AutoML](https://github.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/blob/main/Images/run_details_automl.PNG)

This shows the best model:

![Best model with run ID](https://github.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/blob/main/Images/best_model_with_run_id.PNG)

In the future, I would like to see if letting the experiment run for multiple hours will yield much better results. Also, testing to see how increasing the number of cross-validations or using a larger data set may reduce bias. 


## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Logistic Regression algorithm from the ScikitLearn library was used for the HyperDrive Experiment. These parameters were set up:

<ul>
<li> C determines the strength of the regularization: higher values of C mean less regularization. I used the values (1, 2, 3, 4) </li>
<li> max_iter which is the number of iterations over the entire dataset. I used the values (40, 80, 120, 130, 200)</li>
<li> Random Sampling allowed us optimize resource usage.</li>
</ul>

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?
The HyperDrive experiment yielded an accuracy score of 0.8986 with regularization parameter of 4 and max iteration of 200.

In the future, I would like to test improving the experiment by using a different sampling technique.

The image below shows run details with the run in progress:

![Run details hyperdrive](https://github.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/blob/main/Images/run_details_hyperdrive.PNG)

This shows best run:

![Best run hyperdrive](https://github.com/obinnaonyema/CreditCardChurn_UdacityAZMLCapstone/blob/main/Images/best_run_hyperdrive_with_run_id.PNG)


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
