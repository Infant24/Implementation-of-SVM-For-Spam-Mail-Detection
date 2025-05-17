# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

   
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Infant Maria S tefanie .F
RegisterNumber: 212224230095
*/

import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

### data.head()

![image](https://github.com/user-attachments/assets/c14275b4-bbab-432b-b2ca-535297d2b028)

### data.info()

![image](https://github.com/user-attachments/assets/9f1849b3-f9e1-49d8-ba95-dda3a50d7c1e)

### data.isnull().sum()

![image](https://github.com/user-attachments/assets/c6301f99-5bc9-40ea-8bf4-9d237b8931f4)

### y_prediction value

![image](https://github.com/user-attachments/assets/5a88666b-bac4-4609-83e7-7f4cedcc3701)

### Accuracy value

![image](https://github.com/user-attachments/assets/3475f64b-3f0b-4188-80fc-43f9393f636d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
