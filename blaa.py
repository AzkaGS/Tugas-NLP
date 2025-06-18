import numpy as np
import pandas as pd
import seaborn as sns
import re
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')

'''load training dataset'''
df = pd.read_csv('/content/data.csv')
df.head()

print(df.info())

df['outcome'].value_counts()

# Distribusi Label
plt.figure(figsize=(8, 6))
sns.countplot(x='outcome', data=df)
plt.title('Distribusi Label Berita')
plt.xlabel('Label (0: Fake News, 1: True News)')
plt.ylabel('Jumlah Berita')
plt.show()

# Histogram
df['panjang_teks'] = df['headlines'].apply(len)
plt.figure(figsize=(10, 6))
plt.hist(df['panjang_teks'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram Panjang Teks Berita')
plt.xlabel('Panjang Teks')
plt.ylabel('Jumlah Berita')
plt.show()

# Missing Value
plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

print(df.isnull().sum())

df.isnull().sum()

df['headlines']

import nltk
nltk.download('stopwords')

stem = PorterStemmer()
def stemming(content):
    content = re.sub('[^a-zA-Z]',' ',content)
    content = content.lower()
    content = content.split()
    content = [stem.stem(word) for word in content if not word in stopwords.words('english')]
    content = ' '.join(content)
    return content
df['headlines'] = df['headlines'].apply(stemming)

X = df['headlines'].values
y = df['outcome'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# prompt: buatkan untuk split data 80 dan 20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=2)

sgd = SGDClassifier()
sgd = sgd.fit(X_train,y_train) # Now X_train and y_train are from the same split
y_tpred = sgd.predict(X_train)
y_pred = sgd.predict(X_test)

# SVM Model
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print(f"SVM Accuracy: {svm_accuracy}")
print(f"SVM Classification Report:\n{classification_report(y_test, svm_y_pred)}")

cm = confusion_matrix(y_test, svm_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive']) # Assuming binary classification
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for SVM')
plt.show()

# Logistic Regression Model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_y_pred = logreg_model.predict(X_test)

# Evaluate Logistic Regression
logreg_accuracy = accuracy_score(y_test, logreg_y_pred)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")
print(f"Logistic Regression Classification Report:\n{classification_report(y_test, logreg_y_pred)}")

cm = confusion_matrix(y_test, logreg_y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

print('train score :',accuracy_score(y_train ,y_tpred ))
print('test score :',accuracy_score(y_test , y_pred))
print('con matrix :',confusion_matrix(y_test, y_pred))
print('report :',classification_report(y_test, y_pred ))

con = confusion_matrix(y_test,y_pred)
hmap =sns.heatmap(con,annot=True,fmt="d")
print ('Confusion Matrix',hmap)

labels = np.arange(2)
clf_report = classification_report(y_test,y_pred,labels=labels,target_names=('Fake News','True News'), output_dict=True)
hmap1 = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
print ('Classification Report',hmap1)

fake_news_headlines = ' '.join(df[df['outcome'] == 0]['headlines'])
real_news_headlines = ' '.join(df[df['outcome'] == 1]['headlines'])

fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_news_headlines)
real_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(real_news_headlines)

plt.subplot(2, 1, 1)
plt.imshow(fake_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Fake News Headlines')

plt.tight_layout()
plt.show()

plt.subplot(2, 1, 2)
plt.imshow(real_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Real News Headlines')

plt.tight_layout()
plt.show()

# Create a DataFrame for the comparison
data = {
    'Model': ['SVM', 'Logistic Regression', 'SGD'],
    'Accuracy': [svm_accuracy, logreg_accuracy, accuracy_score(y_test, y_pred)],
    'Precision': [
        classification_report(y_test, svm_y_pred, output_dict=True)['weighted avg']['precision'],
        classification_report(y_test, logreg_y_pred, output_dict=True)['weighted avg']['precision'],
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision']
    ],
    'Recall': [
        classification_report(y_test, svm_y_pred, output_dict=True)['weighted avg']['recall'],
        classification_report(y_test, logreg_y_pred, output_dict=True)['weighted avg']['recall'],
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
    ],
    'F1-score': [

        classification_report(y_test, svm_y_pred, output_dict=True)['weighted avg']['f1-score'],
        classification_report(y_test, logreg_y_pred, output_dict=True)['weighted avg']['f1-score'],
        classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
    ],
    'Vectorizer': ['TF-IDF', 'TF-IDF', 'TF-IDF']  # Assuming TF-IDF is used for all
}
comparison_df = pd.DataFrame(data)

# Display the DataFrame
comparison_df

plt.figure(figsize=(10, 6))
plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Perbandingan Akurasi Testing Model')
plt.xlabel('Model')
plt.ylabel('Akurasi')
plt.ylim(0.8, 1.0)
plt.show()
