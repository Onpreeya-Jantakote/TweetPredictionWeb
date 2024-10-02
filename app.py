from flask import Flask, render_template, request
import joblib
import pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = joblib.load('decision_tree_model.pkl')
le = LabelEncoder()

# Route for the home page
@app.route('/')
def index():
    data = pd.read_csv('vaccination_all_tweets.csv')
    data_50 = data.head(50).to_dict(orient='records')  # Convert to a list of dictionaries

    return render_template('index.html', data_50=data_50)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # รับค่าจากฟอร์ม
        user_name = request.form['user_name']
        user_location = request.form['user_location']
        user_description = request.form['user_description']
        text = request.form['text']
        hashtags = request.form['hashtags']
        retweets = int(request.form['retweets'])
        favorites = int(request.form['favorites'])

        # เข้ารหัสคุณลักษณะประเภทข้อความ
        user_name_enc = le.fit_transform([user_name])[0]
        user_location_enc = le.fit_transform([user_location])[0]
        user_description_enc = le.fit_transform([user_description])[0]
        text_enc = le.fit_transform([text])[0]
        hashtags_enc = le.fit_transform([hashtags])[0]

        # สร้าง DataFrame สำหรับทำพรีดิก
        input_data = pd.DataFrame([[user_name_enc, user_location_enc, user_description_enc, text_enc, hashtags_enc, retweets, favorites]],
                                  columns=['user_name_enc', 'user_location_enc', 'user_description_enc', 'text_enc', 'hashtags_enc', 'retweets', 'favorites'])

        # พรีดิกการรีทวีต
        retweet_prediction = model.predict(input_data)[0]

        # วิเคราะห์ความรู้สึกโดยใช้ TextBlob
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        # กำหนดผลลัพธ์ของการวิเคราะห์ความรู้สึก
        if sentiment > 0:
            sentiment_result = 'Positive'
        elif sentiment < 0:
            sentiment_result = 'Negative'
        else:
            sentiment_result = 'Neutral'

        # โหลดข้อมูลใหม่และแสดง 50 แถวแรก
        data = pd.read_csv('vaccination_all_tweets.csv')
        data_50 = data.head(50).to_dict(orient='records')

        # ส่งผลลัพธ์ของพรีดิกและข้อมูลกลับไปที่เทมเพลต
        return render_template('index.html', sentiment_result=sentiment_result, retweet_prediction=retweet_prediction, data_50=data_50)


if __name__ == '__main__':
    app.run(debug=True)
