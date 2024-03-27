from flask import Flask, render_template, redirect, request
from src.constants import *

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html',
                            stock=STOCK,
                            time_span=DAYS,
                            epochs=EPOCHS)

@app.route('/train_prediction')
def train_prediction():
    return "Hi This training and prediction"

@app.route('/docs')
def docs():
    return 'This is docs part'

if __name__=="__main__":
    app.run(debug=True)