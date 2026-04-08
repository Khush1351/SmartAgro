from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Project initialized"

if __name__ == '__main__':
    app.run(debug=True)