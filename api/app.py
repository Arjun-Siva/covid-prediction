import os
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

print(" * Loading models...")
cnn_model = load_model("models/cnn_model.h5")
feature_selector = Sequential()
for layer in cnn_model.layers[:-2]:
    feature_selector.add(layer)

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
vcf_clf = pickle.load(open('models/vcf_clf.sav', 'rb'))
print(" * Loaded models")

@app.route('/', methods=['GET'])
def index():     
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        img = image.load_img(file_path, target_size=(299,299,3))
        img_arr = (np.expand_dims(image.img_to_array(img), axis=0))/255.0
        fs = feature_selector.predict(img_arr)
        X = scaler.transform(fs)
        result = vcf_clf.predict_proba(X.reshape(1,-1))
        normal, covid = result[0][0], result[0][1]
        # Delete the file
        os.remove(file_path)

        return str(round(normal*100, 2))+","+str(round(covid*100, 2))
    return None

if __name__ == '__main__':
    app.run(debug=False)