import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_tkt= pickle.load(open('model_ttype.pkl', 'rb'))
model_ctg= pickle.load(open('model_category.pkl', 'rb'))
model_imp= pickle.load(open('model_impact.pkl', 'rb'))
model_urg= pickle.load(open('model_urgency.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('Mail_Classifier.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    description = [x for x in request.form.values()]
    final_features = preprocess(description)
    tkt_type = np.argmax(model_tkt.predict(final_features))
    ctg_type = np.argmax(model_ctg.predict(final_features))
    imp_type = np.argmax(model_imp.predict(final_features))
    urg_type = np.argmax(model_urg.predict(final_features))

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)