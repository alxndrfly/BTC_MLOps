# This project's goal is to build a machine learning model capable of predicting Bitcoin's next hour's closing price based on prior data and deploy it with AWS.

## Step 0 : Created a virtual environment and a requirements.txt locally for later deployment

## Step 1 : Connect to a crypto price API and fetch data.

- Used cryptocompare.com
- Fetches past 2000 hours of hourly BTC data

## Step 2 : Train and test 3 different models to compare performance and select the best one.

- Long Short Term Memory : Terrible Results

![image](https://github.com/alxndrfly/BTC_MLOps/assets/135460292/58242c3b-223c-498f-aabf-c60ea51f8dc6)

- XGBoost : Terrible Results

![image](https://github.com/alxndrfly/BTC_MLOps/assets/135460292/d79ebe5b-eccc-4f3e-8a43-2255c4422f1e)

- ARIMA : Best Results

![image](https://github.com/alxndrfly/BTC_MLOps/assets/135460292/4225b50d-dc2b-46af-9dd5-8beb6beb403a)


I selected ARIMA as the model to deploy.

## Step 3 : Add features and prepare for deployment

I wrote two scripts for deployment :

train_model_script.py : 
- Fetches last 2000 hours of price data from the API
- Performs data cleaning and training of the ARIMA model
- Prints the R squared score of the model
- Saves the model as a .pkl file

predict_script.py :
- Loads the model
- Predicts next hour's BTC closing price
- Prints it in a Discord server

## Step 4 : Deploy on AWS

This repo was deployed on a notebook instance of amazon sagemaker.

I added a run_scripts.sh file that runs both python scripts in the right sequence.
I used cron to hourly run run_scripts.sh.

![image](https://github.com/alxndrfly/BTC_MLOps/assets/135460292/bea1b845-d246-4ac9-89c1-12fc63323ce1)


![image](https://github.com/alxndrfly/BTC_MLOps/assets/135460292/42c2f5a1-3a61-49f1-bd90-aae5f4b49530)
