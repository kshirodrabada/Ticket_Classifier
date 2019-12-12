import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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
    description_tokens = tokenizer.texts_to_sequences(description)
    final_features = pad_sequences(description_tokens, maxlen = 1000)    
    tkt_type = np.argmax(model_tkt.predict(final_features))
    ctg_type = np.argmax(model_ctg.predict(final_features))
    imp_type = np.argmax(model_imp.predict(final_features))
    urg_type = np.argmax(model_urg.predict(final_features))    

    return render_template('Mail_Classifier.html', prediction_text=tkt_type)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    
    return jsonify(tkt_type)

if __name__ == "__main__":
    app.run(debug=True)
