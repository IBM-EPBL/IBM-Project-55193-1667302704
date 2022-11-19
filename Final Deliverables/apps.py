from flask import Flask, render_template,url_for, request
import pickle
import numpy as np

model = pickle.load(open('wet.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('final.html')


@app.route('/predict', methods=['GET','POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    
    

    
if __name__ == "__main__":
    app.run(debug=True)


