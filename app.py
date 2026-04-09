from flask import Flask, render_template, request, Markup
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import pickle
import disease_cnn as disease_CNN
import torchvision.transforms.functional as TF
import torch
from PIL import Image

classifier = load_model('Trained_model.h5')
classifier.make_predict_function()

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

disease_info      = pd.read_csv('Data/disease_info.csv',    encoding='cp1252')
supplement_info   = pd.read_csv('Data/supplement_info.csv', encoding='cp1252')

disease_model = disease_CNN.CNN(39)
disease_model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
disease_model.eval()

def predict_disease(image_path):
    image      = Image.open(image_path)
    image      = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output     = disease_model(input_data)
    output     = output.detach().numpy()
    index      = np.argmax(output)
    return index

app = Flask(__name__)

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)


def pred_pest(pest):
    try:
        test_image = load_img(pest, target_size=(64, 64))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(classifier.predict(test_image), axis=-1)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return 'x'

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)

        if pred == 'x':
            return render_template('unaptfile.html')
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')

@app.route("/DiseaseHome.html")
def disease_home():
    return render_template("DiseaseHome.html")
 
@app.route("/DiseaseDetection.html")
def disease_detection():
    return render_template("DiseaseDetection.html")
 
 
@app.route("/disease-predict", methods=['GET', 'POST'])
def disease_predict():
    if request.method == 'POST':
        image     = request.files['image']
        filename  = image.filename
        file_path = os.path.join('static/user uploaded', filename)
        image.save(file_path)
 
        pred             = predict_disease(file_path)
        title            = disease_info['disease_name'][pred]
        description      = disease_info['description'][pred]
        prevent          = disease_info['Possible Steps'][pred]
        image_url        = disease_info['image_url'][pred]
        supplement_name  = supplement_info['supplement name'][pred]
        supplement_image = supplement_info['supplement image'][pred]
        buy_link         = supplement_info['buy link'][pred]
 
        return render_template('DiseaseResult.html',
                               title=title,
                               desc=description,
                               prevent=prevent,
                               image_url=image_url,
                               pred=pred,
                               sname=supplement_name,
                               simage=supplement_image,
                               buy_link=buy_link,
                               uploaded_image=file_path)

if __name__ == '__main__':
    app.run(debug=True)
