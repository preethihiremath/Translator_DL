# Import libraries
from flask import Flask, render_template, redirect, request
import helper as hlp

app = Flask(__name__)

# fetch the model api
@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == "POST":
        input_text = request.form['input_lang_text']
        translated_text= hlp.predict(input_text)
        return render_template("landing.html",translated_text=translated_text, input_text=input_text)
    return render_template("landing.html")

if __name__ == '__main__':
    app.run(port=5000, debug=True)
