# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# Importing dfs
df = pd.read_csv('ISMDatasetSentiment.csv', encoding='ISO-8859-1')
print(df.sample(5))
print(df.shape)
print(df.head())

# Checking Null values
print(df.isnull().sum())

# Checking duplicate values
print(df.duplicated().sum())

# Checking Churn Values
print(df['Churn'].value_counts())

# Plotting the values
plt.pie(df['Churn'].value_counts(), labels=['not churn', 'churn'], autopct="%0.2f")
plt.show()

# Encoding gender to binary value
lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])
print(df.head())

# Adding Sentiment analysis to get sentiment score
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def get_sentiment_score(text):
    if len(text) > 512:
        text = text[:510]
    try:
        result = sentiment_pipeline(text)[0]
        score = round(result['score'], 2)
        if result['label'] == 'LABEL_0':
            score *= -1
        return score
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return None

sample_texts = [
    "I love this product, it's amazing!",
    "This movie was terrible, I hated it.",
    "The weather today is perfect.",
    "I'm feeling neutral about this situation.",
]
results = list(map(get_sentiment_score, sample_texts))
for text, score in zip(sample_texts, results):
    print(f"Text: {text}")
    print(f"Sentiment Score: {score}")
    print()

df.loc[df['Sentiment_Score'].isnull(), 'Sentiment_Score'] = df[df['Sentiment_Score'].isnull()]['Summary'].apply(get_sentiment_score)

# Visual Feature Description
numerical_features = ['Tenure', 'AppTime', 'OrderIncrease', 'OrderCount', 'InactiveDays', 'CashbackAmount', 'Sentiment_Score']
df[numerical_features].hist(bins=20, figsize=(15, 10))
plt.show()

# Explore Categorical Features
plt.figure(figsize=(10, 5))
sns.countplot(x='Gender', hue='Churn', data=df)
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='Complain', hue='Churn', data=df)
plt.show()

# Correlation analysis
correlation_matrix = df[numerical_features].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.show()

# Feature Relationships
plt.figure(figsize=(12, 8))
sns.boxplot(x='Churn', y='Tenure', data=df)
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='OrderCount', y='Sentiment_Score', hue='Churn', data=df)
plt.show()

# Text Analysis
text = " ".join(review for review in df.Text)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Feature Engineering
df['OrderRatio'] = df['OrderCount'] / df['OrderIncrease']
df.drop(columns=["Summary", "Text"], inplace=True)

# Feature Importance
X = df.drop('Churn', axis=1)
y = df['Churn']
model = RandomForestClassifier()
model.fit(X, y)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.show()

#Changing Datatpe of sentiment scorefrom object to float
df['Sentiment_Score'] = pd.to_numeric(df['Sentiment_Score'], errors='coerce')
print(df.dtypes)

# Model Building
# Assuming X contains the features and y contains the target variable (Churn)

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the df into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#XGBoost Classifier
# Assuming X contains the features and y contains the target variable (Churn)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the df into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Create a DataFrame from y_test and y_pred
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Calculate the churn and non-churn counts for actual and predicted
comparison = pd.DataFrame({
    'Actual Churn': results_df['Actual'].value_counts(),
    'Predicted Churn': results_df['Predicted'].value_counts()
})

print(comparison)

#Graph Plotting

#1. Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Churn Status', alpha=0.5)
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Churn Status', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('Churn Status')
plt.title('Actual vs. Predicted Churn Status')
plt.legend()
plt.show()

# 2. Comparison Bar Graph
# Plotting the comparison
comparison.plot(kind='bar')
plt.title('Comparison of Actual and Predicted Churn')
plt.xlabel('Churn Status')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'], rotation=0)  # Adjust labels according to your specific labels if different
plt.legend()
plt.show()

#3. Confusion Chart

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

#4. ROC Curve
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # Random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
average_precision = average_precision_score(y_test, model.predict_proba(X_test)[:,1])

plt.figure()
plt.step(recall, precision, where='post', label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

#Bar Graph of Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

df_report.drop(['accuracy'], inplace=True)  # drop the total accuracy since it's a single number
df_report['support'] = df_report['support'].apply(int)  # convert from float to int

df_report[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title('Classification Report')
plt.xlabel('Classes')
plt.ylabel('Scores')
plt.xticks(rotation=0)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


def predict_churn(data):
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, classification_rep, conf_matrix