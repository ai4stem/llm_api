from flask import (
    Flask, request, render_template, url_for, redirect, flash, session, jsonify)
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

@app.route('/')
def hello():
    return "It is really drizzle and windy!!"


@app.route('/greeting')
def greeting():
    return "Hello, Dear."
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

