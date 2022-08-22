# Credit Scoring Using Logistic Regression
As a data science intern at ID/X Partners (Virtual Internship Program), I was assigned to make a credit risk analysis from lending company. My objective here is to make a good prediction model that can clasify whether borrowers are good or not. 

I used logistic regression as a model to predict credit risk and implemented weight evidence & information values to perfrom feature selection

Metrics used in this project are ROC-AUC score and KS-Statistic

My goals here are:
1. build a prediction model with good ROC-AUC score (>0.8) and good KS-Statistic score (>0.45) 
2. build credit score (score card) each borrower and treshold recommendation list

## Feature Engineering and Feature Selection
Before implementing weight of evidence and information value method, we have 27 features consist of 4 categorical features and 23 numeric features. We combined features that have similar weight of evidence and dropped those who didn't follow rule of thumb. After implementing selection method, now we have dataset that consist of 466k records and   85 features


## Modeling
We used  logistic regression to predict whether the borrower will be a good  borrower or not.  We also used AUC score and KS-Statistic as our evaluation metric. After performing hyperparameter tuning and feature selection, the results are :
1. AUC Training Score : 0.847
2. AUC Testing Score : 0.848
3. KS Statistic : 0.552


![Screen Shot 2022-08-22 at 15 58 45](https://user-images.githubusercontent.com/106853320/185882297-32e802dc-4788-4c30-9055-3f71ab716665.png

## Feature Importance:
1. Last Payment:5 Month
Borrowers who made the last payment in the past 5 months, their odds of being a good loan borrowers will increase by 1.9 times
2. Last Payment:5-7 Month 
Borrowers who made the last payment in the past 5-7 months, their odds of being a good loan borrowers will increase by 1.7 times
3.Interest Rate: 5.3%-7.4%
Borrowers with an interest rate of 5.3%-7.4%, their odds of being a good loan borrowers will increase by 1.5 times

## Credit Score Card:

![Screen Shot 2022-08-22 at 16 00 17](https://user-images.githubusercontent.com/106853320/185882656-922738df-3beb-471e-b483-80a0d251eaf2.png)

full credit score file: [Download](https://drive.google.com/file/d/1vTWB1ZD-dyQJE9BI4Ssl4KkBqHQq26C4/view?usp=sharing)

![Screen Shot 2022-08-22 at 16 00 24](https://user-images.githubusercontent.com/106853320/185882689-03fd8599-004d-41d5-8f48-162cc301a26f.png)

full credit score file: [Download](https://drive.google.com/file/d/1vOk4zGgx2iKROi6ySVn_Devt5Hl39Ijf/view?usp=sharing)

## Insights:

![figure11](https://user-images.githubusercontent.com/106853320/185883093-ef6955fb-d334-4378-b245-696b8f84fade.png)

![figure22](https://user-images.githubusercontent.com/106853320/185883128-81f05c35-18f9-4134-958f-6d931525b3cb.png)

![figure33](https://user-images.githubusercontent.com/106853320/185883153-6683c319-ad42-4095-a0f7-067044dee3eb.png)



