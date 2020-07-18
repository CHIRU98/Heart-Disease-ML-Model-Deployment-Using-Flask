from flask import Flask,request,render_template
import pickle
import numpy as np

model = pickle.load(open('model.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == [1]:
        return render_template('index.html', prediction_text='Yess!! You Have Disease {}'.format(prediction))
    elif prediction == [0]:
        return render_template('index.html', prediction_text='Noo!! You Dont Have Disease {}'.format(prediction))



app.run(debug=True)


