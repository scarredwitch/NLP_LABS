from flask import Flask, render_template, request
from utils.general import load_lstm,generate   
from torchtext.data.utils import get_tokenizer
#Load GPU
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import DataRequired
import torch
device = torch.device('cpu')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')   
import torch
import torchtext

app = Flask(__name__, template_folder='template')             # create an app instance
app.config['SECRET_KEY'] = 'mykey'
app.config['UPLOAD_FOLDER'] = 'static/files' 


@app.route('/')
def index():
    return render_template('index.html')

class MyForm(FlaskForm):
    name = StringField('Try it out !!', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/autocomplete', methods = ['GET','POST'])
def autocomplete():
    form = MyForm()
    code = False
    name = False
    print(form.validate_on_submit())
    model,vocab_dict = load_lstm()
    if form.validate_on_submit():
        name = form.name.data 
        temperature =  [0.4]
        code = [' '.join(generate(name.strip(), 30, temp, model, tokenizer, vocab_dict, device, seed=0)) for temp in temperature]
        form.name.data = ""
    return render_template("autocomplete.html",form=form,name =name, code=code)



if __name__ == "__main__":        # on running python app.py
    app.run(debug=True)    