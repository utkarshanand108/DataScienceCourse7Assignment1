# DataScienceCourse7_Assignment1
**Text Classification using Logistic Regression — DS PGC Course 7 Assignment 1**

---

## 📘 Assignment Overview
This assignment introduces **Natural Language Processing (NLP)** fundamentals by building a simple **text classification model** using Python.  
You learn to clean, tokenize, and vectorize text data, and train a **Logistic Regression** classifier to predict sentiment labels.

---

## 🧩 Tasks Summary

### 🧠 Task 1 — Data Exploration
- Loaded dataset: `text_class.csv`
- Displayed first 5 rows and identified class distribution.

```python
import pandas as pd
df = pd.read_excel('text_class.xlsx')
print(df.head())
print(df['label'].value_counts())
```

**Dataset summary:**  
| Label | Count |
|:------|:-------|
| positive | 3 |
| negative | 3 |
| neutral | 2 |
| **Total rows:** | **8** |

---

### 🧹 Task 2 — Text Preprocessing
Steps:
1. Convert text to lowercase  
2. Remove punctuation & special characters  
3. Tokenize and remove stopwords  
4. Join tokens back to clean sentences  

```python
import re, nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)
print(df[['text', 'clean_text']].head())
```

**Result (first 5 cleaned rows):**  
| Original Text | Cleaned Text |
|:---------------|:-------------|
| I loved the product, it's amazing! | loved product amazing |
| Terrible service, I will never shop here again. | terrible service never shop |
| The quality is good, but delivery was late. | quality good delivery late |

---

### 🔢 Task 3 — Train a Classifier
- Converted text to numeric features using **TF-IDF Vectorizer**  
- Split data into **80% training / 20% testing**  
- Trained a **Logistic Regression** model  

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

**Model accuracy:** `0.0` (due to very small dataset)  

> ⚠️ The dataset has only 8 samples — not enough for meaningful training, showing why larger datasets are crucial in NLP.

---

### 📊 Task 4 — Model Evaluation
Used a **confusion matrix** to analyze predictions:

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

**Confusion Matrix:**
```
[[0 1 0]
 [0 0 0]
 [0 1 0]]
```

> The confusion matrix shows how many predictions were correct or incorrect for each class.  
It helps visualize specific misclassifications instead of just relying on accuracy.

---

## 🧰 Tools & Techniques
| Category | Details |
|:-----------|:-----------|
| Language | Python 3 |
| Environment | Jupyter Notebook / Google Colab |
| Libraries | pandas, numpy, scikit-learn, nltk, re |
| Model | Logistic Regression |
| Techniques | Text Cleaning, Tokenization, Stopword Removal, TF-IDF, Train-Test Split |

---

## 📂 Files Included
```
TextClassification.pdf                     # Problem statement
TextClassificationPDF.pdf                  # Solution report (PDF)
TextClassificationPythonNotebook.ipynb     # Jupyter Notebook solution
TextClassificationPythonScript.py          # Python script version
```

---

## 🧭 How to Review
1. Open `.ipynb` for code and output visualization.  
2. View `.py` for clean executable code.  
3. Read the PDF for summary and explanations.  
4. Dataset link (Google Sheets) in the problem statement.

---

## 👤 Author
**Utkarsh Anand**  
Data Science PGC Course 7 — Assignment 1  
Internshala Placement Guarantee Program
