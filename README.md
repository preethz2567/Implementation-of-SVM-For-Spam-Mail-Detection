# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program / Output:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PREETHI D
RegisterNumber: 212224040250
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
```
df = pd.read_csv('spam.csv', encoding='latin-1')
print(df.head())
```
![image](https://github.com/user-attachments/assets/b410844f-4321-4a01-b7e8-f21eada12fdf)

```
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
```
```
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
```
```
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['message'])
y = df['label_num']
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```
model = SVC(kernel='linear')  # linear kernel works well for text classification
model.fit(x_train, y_train)
```
![image](https://github.com/user-attachments/assets/10c3c603-8564-44ad-924e-18935d86207f)

```
y_pred = model.predict(x_test)
```
```
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
![image](https://github.com/user-attachments/assets/6947038a-438d-44cd-897f-6b899ab84d6d)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
