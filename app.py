from datetime import datetime
from flask import Flask, render_template, url_for, request, redirect,make_response

from flask_sqlalchemy import SQLAlchemy

from val import Val
from verification import Verify
import os


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///skoltech.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    intro = db.Column(db.String(300), nullable=False)
    text = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Article %r>' & self.id


@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/classification')
def clf():
    return render_template("classification.html")


@app.route('/verification')
def verification():
    return render_template("verification.html")


@app.route('/contact', methods=['POST', 'GET'])
@app.route('/record', methods=['POST', 'GET'])
def record():
    if request.method == "POST":
        title = request.form['title']
        intro = request.form['intro']
        text = request.form['text']
        answer = 'no'

        article = Article(title=title, intro=intro, text=text, answer=answer)

        try:
            db.session.add(article)
            db.session.commit()
            return redirect('/faq')
        except:
            return "Error while creating request"
    else:
        return render_template("contact.html")


@app.route('/qanda')
def faq():
    articles = Article.query.order_by(Article.date.desc()).all()
    return render_template("qanda.html", articles=articles)



@app.route('/qanda/<int:id>', methods=['POST', 'GET'])
def add_answer(id):
    article = Article.query.get(id)
    if request.method == "POST":
        article.answer = request.form['answer']

        try:
            db.session.commit()
            return redirect('/faq')
        except:
            return "Error while adding answer"
    else:
        return render_template("add_answer.html", article=article)

'''
@app.route('/classification/results')
def clf_res():
    return render_template("classification_results.html")
'''


@app.route('/results_classification', methods=['POST', 'GET'])
def asr():
    if request.method == "POST":
        request_file = request.files['myfile']
        #print((request_file))
        if not request_file:
            return "No file"

        predictions_gender, predictions_age, predictions_emotion = Val().predict(request_file)

        if predictions_gender == "male":
            predictions_gender = "Мужской"
        else:
            predictions_gender = "Женский"

        # if predictions_gender == "teens":
        #     predictions_age = "Подросток"
        # elif predictions_gender == "twenties":
        #     predictions_age = "Двадцать лет"
        # elif predictions_gender == "thirties":
        #     predictions_age = "Тридцать лет"
        # else:
        #     predictions_age = "Больше сорока"


        res = {
            'gender': predictions_gender,
            'age': predictions_age,
            'emotion': predictions_emotion,
        }

       # print(res['gender'])
        #response = make_response(f
        #response.headers["Content-Disposition"] = "attachment; filename=result.txt"

    return render_template("classification_results.html", title='Edit Creative', result=res)


@app.route('/results_verification', methods=['POST', 'GET'])
def verf():
    if request.method == "POST":
        save_path1 = os.path.join('./', "wav1.wav")
        save_path2 = os.path.join('./', "wav2.wav")

        request_file_1 = request.files['myfile'].save(save_path1)
        request_file_2 = request.files['myfile1'].save(save_path2)



        answer = Verify().predict(save_path1, save_path2)
        #print(answer)
        if answer[1] == False:
            ans = "No, it is not the same person"

        elif answer[1] == True:
            ans = "Yes, it is the same person"
        res = {"result": ans}
        os.remove(save_path1)
        os.remove(save_path2)
        return render_template("verification_results.html", title='Edit Creative', result=res)

# @app.route('/classification/<path:filename>', methods=['GET'])
# def download_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
