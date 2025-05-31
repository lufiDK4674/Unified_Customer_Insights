from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
import seaborn as sns
from churnprediction import predict_churn

app = Flask(__name__)
CORS(app) 
sentiment_pipeline = pipeline("sentiment-analysis" , model = "cardiffnlp/twitter-roberta-base-sentiment")

label_mapping = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive'
}
def score_to_rank(label, score):
    if label == 'Positive':
        if score > 0.7:
            return 'High'
        elif score > 0.5:
            return 'Medium'
        else:
            return 'Low'
    elif label == 'Negative':
        if score > 0.7:
            return 'Low'
        elif score > 0.5:
            return 'Medium'
        else:
            return 'High'
    else:
        return 'N/A'
    
# #Churn Prediction
# # Importing dfs
# df = pd.read_csv('ISMDatasetSentiment.csv', encoding='ISO-8859-1')
# print(df.sample(5))
# print(df.shape)
# print(df.head())

# # Encoding gender to binary value
# lb = LabelEncoder()
# df['Gender'] = lb.fit_transform(df['Gender'])
# print(df.head())

# # Adding Sentiment analysis to get sentiment score
# sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# def get_sentiment_score(text):
#     if len(text) > 512:
#         text = text[:510]
#     try:
#         result = sentiment_pipeline(text)[0]
#         score = round(result['score'], 2)
#         if result['label'] == 'LABEL_0':
#             score *= -1
#         return score
#     except Exception as e:
#         print(f"Error processing text: {text}, Error: {e}")
#         return None

# df.loc[df['Sentiment_Score'].isnull(), 'Sentiment_Score'] = df[df['Sentiment_Score'].isnull()]['Summary'].apply(get_sentiment_score)

# # Feature Engineering
# df['OrderRatio'] = df['OrderCount'] / df['OrderIncrease']
# df.drop(columns=["Summary", "Text"], inplace=True)

# #Changing Datatpe of sentiment scorefrom object to float
# df['Sentiment_Score'] = pd.to_numeric(df['Sentiment_Score'], errors='coerce')
# print(df.dtypes)

# #XGBoost Classifier
# # Assuming X contains the features and y contains the target variable (Churn)
# X = df.drop('Churn', axis=1)
# y = df['Churn']

# # Split the df into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the XGBoost model
# model = xgb.XGBClassifier()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# # Print classification report and confusion matrix
# print(classification_report(y_test, y_pred))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# # Create a DataFrame from y_test and y_pred
# results_df = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred
# })
# # Calculate the churn and non-churn counts for actual and predicted
# comparison = pd.DataFrame({
#     'Actual Churn': results_df['Actual'].value_counts(),
#     'Predicted Churn': results_df['Predicted'].value_counts()
# })

# print(comparison)
    
# @app.route('/')
# def home():
#     return 'Python Server running'

# @app.route('/predict_sentiment', methods=['POST'])
# def predict_sentiment():
#     data = request.json
#     text = data['text']
#     result = sentiment_pipeline(text)
#     sentiment_label = label_mapping[result[0]['label']]
#     sentiment_score = result[0]['score']
#     sentiment_score_rank = score_to_rank(sentiment_label, sentiment_score)
#     return jsonify({
#         'sentiment': sentiment_label,
#         'score_rank': sentiment_score_rank
#     })
    
# @app.route('/predict_churn', methods=['POST'])
# def predict_churn_endpoint():
#     try:
#         data = request.json
#         accuracy, classification_rep, conf_matrix = predict_churn(data)
#         return jsonify({
#             'accuracy': accuracy,
#             'classification_report': classification_rep,
#             'confusion_matrix': conf_matrix.tolist()
#         })
#     except Exception as e:
#         # Handle any errors
#         return jsonify({'error': str(e)}), 500
        

# if __name__ == '__main__':
#     app.run(debug=True)
