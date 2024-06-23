Welcome to our project.

This repo was deployed on a notebook instance of amazon sagemaker.
I used cron to periodically run the python scripts.
The scripts fetch hourly BTC price data from the crypto compare API and train the model on the past 2000 hours of hourly BTC prices. 

Then they generate and save the model and use the model to predict the next hourly closing price.

The prediction message then gets sent to a discord server for users to see.
