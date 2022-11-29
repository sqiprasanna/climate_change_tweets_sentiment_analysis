### Flask

from flask import Flask
from flask import request, render_template
import json
from .model import sent_analysis

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")
    # return "<p>Hello, World!</p>"


@app.route('/sentiment', methods=['POST'])
def sentiment():
    print("Got a request")
    print(request.json)
    tweet = request.json['Tweet']
    print(sent_analysis(tweet)[0])
    return {
        "Sentiment": sent_analysis(tweet)[0],
    }
    # return getResult(data, time)