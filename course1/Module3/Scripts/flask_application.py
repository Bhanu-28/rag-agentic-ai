from flask import Flask

app = Flask("Bhanu Web Application")

@app.route('/')
def hello_World():
    return "Hello Bhanu Pradeep welcome!"


if __name__ == '__main__':
    app.run(debug=True)