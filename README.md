# 🧠 Sentiment Analysis of Restaurant Reviews using Naive Bayes

---

## 1️⃣ Project Overview

This project builds a sentiment classifier to determine whether a restaurant review expresses **positive** or **negative** sentiment. The approach involves a complete Natural Language Processing (NLP) pipeline: text preprocessing, Bag-of-Words feature extraction, and classification using **Multinomial Naive Bayes**. Despite its simplicity, this method performs surprisingly well on short text data.

---

## 2️⃣ Dataset Info

- **Source**: [Kaggle – Restaurant Reviews Dataset](https://www.kaggle.com/datasets/tsuaravi/restaurant-reviews)  
- **Samples**: 1000 restaurant reviews  
- **Format**: `.tsv` (Tab-Separated Values)  
- **Features**:
  - `Review`: Text of the customer review
  - `Liked`: Sentiment label — `1` for positive, `0` for negative

---

## 3️⃣ Data Preprocessing

Each review undergoes the following preprocessing steps:
- Remove non-alphabet characters (e.g., punctuation and digits) using `regex`
- Convert all text to lowercase
- Tokenize each sentence into words
- Remove common English stopwords (using `nltk.corpus.stopwords`)
- Apply **stemming** to reduce words to their root form (using `PorterStemmer`)
- Join the cleaned words back into a single string

The cleaned reviews are stored in a list called `corpus`, which is later passed to the vectorizer.

---

## 4️⃣ Model

### ✳️ Vectorization
- **Tool**: `CountVectorizer` from `sklearn`
- **Method**: Bag-of-Words (BoW)
- **Vocabulary Size**: 1500 most frequent words
- **Output**:  
  - `X`: 1000 × 1500 sparse matrix of word counts  
  - `y`: 1D array of sentiment labels

### ✳️ Classifier
- **Algorithm**: `MultinomialNB` (Multinomial Naive Bayes)
- **Why**: Ideal for discrete count features like word frequencies
- **Training**: Done using `classifier.fit(X_train, y_train)`

The model learns the probability of each word occurring in positive and negative reviews and uses **Bayes’ theorem** with log-likelihood to classify new reviews.

---

## 5️⃣ Evaluation

Used a standard **80/20 train-test split**.

### 🔍 Metrics
- **Accuracy Score**: 76.5 %
- **Confusion Matrix**
      TN - 72
      TP - 81
      FP - 25
      FN - 22

This shows good performance with relatively low misclassification.

---

## 6️⃣ Conclusion

This project shows that even a basic NLP pipeline using:
- Proper text cleaning
- Bag-of-Words representation
- Multinomial Naive Bayes classification

...can yield solid performance on sentiment classification tasks. It is an excellent beginner-friendly model to understand the full flow of NLP projects.

---

## 7️⃣ Next Steps 🚀

To improve and extend this project further:
- 🔁 **Upgrade to TF-IDF**: Use `TfidfVectorizer` for better word weighting
- ⚙️ **Try other classifiers**: Logistic Regression, SVM, or Random Forest
- 📉 **Analyze misclassifications**: Look at false positives/negatives to understand errors
- 🤖 **Use pretrained models**: Experiment with transformers like BERT for deep language understanding

---


