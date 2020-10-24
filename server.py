from flask import Flask, request, render_template,flash,redirect,session,abort,jsonify
from models import Model
from depression_detection_tweets import DepressionDetection
from TweetModel import process_message
import os

app = Flask(__name__)


@app.route('/')
def root():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')


@app.route('/login', methods=['POST'])
def do_admin_login():
    if request.form['password'] == 'admin' and request.form['username'] == 'admin':
        session['logged_in'] = True
    else :
        flash('wrong password!')
    return root()

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return root()


@app.route("/sentiment")
def sentiment():
    return render_template("sentiment.html")


@app.route("/predictSentiment", methods=["POST"])
def predictSentiment():
    message = request.form['form10']
    pm = process_message(message)
    result = DepressionDetection.classify(pm, 'bow') or DepressionDetection.classify(pm, 'tf-idf')
    return render_template("tweetresult.html",result=result)


@app.route('/predict', methods=["POST"])
def predict():
    q1 = int(request.form['a1'])
    q2 = int(request.form['a2'])
    q3 = int(request.form['a3'])
    q4 = int(request.form['a4'])
    q5 = int(request.form['a5'])
    q6 = int(request.form['a6'])
    q7 = int(request.form['a7'])
    q8 = int(request.form['a8'])
    q9 = int(request.form['a9'])
    q10 = int(request.form['a10'])

    values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    model = Model()
    classifier = model.svm_classifier()
    prediction = classifier.predict([values])
    if prediction[0] == 0:
            result = 'Your Depression test result : No Depression'
    if prediction[0] == 1:
            result = 'Your Depression test result : Mild Depression'
    if prediction[0] == 2:
            result = 'Your Depression test result : Moderate Depression'
    if prediction[0] == 3:
            result = 'Your Depression test result : Moderately severe Depression'
    if prediction[0] == 4:
            result = 'Your Depression test result : Severe Depression'
    return render_template("result.html", result=result)

app.secret_key = os.urandom(12)
app.run(port=5987, host='0.0.0.0', debug=True)