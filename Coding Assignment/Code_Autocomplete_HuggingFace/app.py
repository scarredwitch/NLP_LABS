from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import pandas as pd

import torch
from transformers import pipeline

pipe = pipeline("text-generation", max_length=30, pad_token_id=0, model="scarredwitch/codeparrot-gpt2-finetune") #eos_token_id=0

app = Flask(__name__,template_folder='template')
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 

@app.route('/')
def index():
    return render_template("index.html")



class MyForm(FlaskForm):
    text = StringField('Type something', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/autocomplete')
def autocomplete():
    form = MyForm()
    gen = ""
    text = ""

    source = request.form.get('source')
    print("...........",source)
    gen = pipe(text)[0]["generated_text"]
    print(gen)
    form.text.data = source
    data = {"source":source, "predict":gen}
    print(data)
    return render_template("autocomplete.html",data= data)

if __name__ == "__main__":
    app.run(debug=True)