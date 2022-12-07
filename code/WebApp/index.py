### Flask

from flask import Flask
from flask import request, render_template
import json
from .model import sent_analysis

app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/sentiment', methods=['POST'])
def sentiment():
    print("Got a request")
    print(request.json)
    tweet = request.json['Tweet']
    res = sent_analysis(tweet)
    print("Response",res)
    return {
        "Sentiment": res[0],
        "Message": res[0],
    }
   