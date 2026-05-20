#!python
import pickle
from flask import Flask, request
#from flask import flasgger
from flasgger import Swagger
import numpy as np
import pandas as pd
import json


with open('GLM.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict')
def predict():
    """Take input and predict business outcome as Event vs Not Event
    ---
    parameters:
      - name: input_json
        in: query
        type: string
        required: true

    responses:
      200:
        description: Predicting Business Outcome
      
    """

    
    input_json = request.args.get("input_json")
    json_data = json.loads(input_json)
    print(type(json_data))
    input_json_data = pd.DataFrame([json_data])
    input_json_data = input_json_data.transpose()
    print(input_json_data)

    input_data = pd.DataFrame(json_data)

    df = input_data.filter(['x5', 'x81', 'x31', 'x12', 'x44', 'x53', 'x56', 'x58', 'x62', 'x91'], axis=1)

    df['x12'] = df['x12'].str.replace('$', '')
    df['x12'] = df['x12'].str.replace(',', '')
    df['x12'] = df['x12'].str.replace(')', '')
    df['x12'] = df['x12'].str.replace('(', '-')
    df['x12'] = df['x12'].astype(float)

    data = {'x5_saturday': 0,
            'x81_July': 0,
            'x81_December': 0,
            'x31_japan': 0,
            'x81_October': 0,
            'x5_sunday': 0,
            'x31_asia': 0,
            'x81_February': 0,
            'x91': 0,
            'x81_May': 0,
            'x5_monday': 0,
            'x81_September': 0,
            'x81_March': 0,
            'x53': 0,
            'x81_November': 0,
            'x44': 0,
            'x81_June': 0,
            'x12': 0,
            'x5_tuesday': 0,
            'x81_August': 0,
            'x81_January': 0,
            'x62': 0,
            'x31_germany': 0,
            'x58': 0,
            'x56': 0}

    ind = np.arange(len(df))

    predict_data = pd.DataFrame(data, index=ind)
    predict_data['x91'] = df['x91']
    predict_data['x53'] = df['x53']
    predict_data['x44'] = df['x44']
    predict_data['x12'] = df['x12']
    predict_data['x62'] = df['x62']
    predict_data['x58'] = df['x58']
    predict_data['x56'] = df['x56']

    predict_data["x5_saturday"] = np.where(df["x5"] == "saturday", 1, 0)
    predict_data["x5_monday"] = np.where(df["x5"] == "monday", 1, 0)
    predict_data["x81_December"] = np.where(df["x81"] == "December", 1, 0)
    predict_data["x81_July"] = np.where(df["x81"] == "July", 1, 0)
    predict_data["x31_japan"] = np.where(df["x31"] == "japan", 1, 0)
    predict_data["x81_October"] = np.where(df["x81"] == "October", 1, 0)
    predict_data["x5_sunday"] = np.where(df["x5"] == "sunday", 1, 0)
    predict_data["x31_asia"] = np.where(df["x31"] == "asia", 1, 0)
    predict_data["x81_February"] = np.where(df["x81"] == "February", 1, 0)

    predict_data["x81_May"] = np.where(df["x81"] == "May", 1, 0)
    predict_data["x5_monday"] = np.where(df["x5"] == "monday", 1, 0)
    predict_data["x81_September"] = np.where(df["x81"] == "September", 1, 0)
    predict_data["x81_March"] = np.where(df["x81"] == "March", 1, 0)

    predict_data["x81_November"] = np.where(df["x81"] == "November", 1, 0)

    predict_data["x81_June"] = np.where(df["x81"] == "June", 1, 0)

    predict_data["x5_tuesday"] = np.where(df["x5"] == "tuesday", 1, 0)
    predict_data["x81_August"] = np.where(df["x81"] == "August", 1, 0)
    predict_data["x81_January"] = np.where(df["x81"] == "January", 1, 0)

    predict_data["x31_germany"] = np.where(df["x31"] == "germany", 1, 0)

    print(predict_data.shape)
    prediction = model.predict(predict_data)
    outcome = pd.DataFrame(prediction).rename(columns={0: 'probs'})

    outcome['Business_Outcome'] = ["Event" if x >= 0.75 else "Not Event" for x in outcome['probs']]

    all_predictions = outcome.copy()

    all_predictions.rename(columns={'probs': 'phat'}, inplace=True)

    all_predictions['model_input'] = input_json_data[0].astype('string')
    all_predictions['model_input'] = all_predictions['model_input'].astype('string')

    #print(str(all_predictions.to_json(orient='records', lines = True)))

    return str(all_predictions.to_json(orient='records'))  # str(final_prediction) #str(list(prediction))


@app.route('/predict_file', methods=["POST"])
def predict_glm_file():
    """Take input as file and predict business outcome as Event vs Not Event
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true

    responses:
      200:
        description: Predicting Business Outcome
    """

    input_json = pd.DataFrame([pd.read_json('test_case.json', typ='series')])

    input_json = input_json.transpose()

    input_json_tolist = input_json[0].tolist()

    input_data = pd.DataFrame(input_json_tolist)

    df = input_data.filter(['x5', 'x81', 'x31', 'x12', 'x44', 'x53', 'x56', 'x58', 'x62', 'x91'], axis=1)

    df['x12'] = df['x12'].str.replace('$', '')
    df['x12'] = df['x12'].str.replace(',', '')
    df['x12'] = df['x12'].str.replace(')', '')
    df['x12'] = df['x12'].str.replace('(', '-')
    df['x12'] = df['x12'].astype(float)

    data = {'x5_saturday': 0,
            'x81_July': 0,
            'x81_December': 0,
            'x31_japan': 0,
            'x81_October': 0,
            'x5_sunday': 0,
            'x31_asia': 0,
            'x81_February': 0,
            'x91': 0,
            'x81_May': 0,
            'x5_monday': 0,
            'x81_September': 0,
            'x81_March': 0,
            'x53': 0,
            'x81_November': 0,
            'x44': 0,
            'x81_June': 0,
            'x12': 0,
            'x5_tuesday': 0,
            'x81_August': 0,
            'x81_January': 0,
            'x62': 0,
            'x31_germany': 0,
            'x58': 0,
            'x56': 0}

    ind = np.arange(len(df))

    predict_data = pd.DataFrame(data, index=ind)
    predict_data['x91'] = df['x91']
    predict_data['x53'] = df['x53']
    predict_data['x44'] = df['x44']
    predict_data['x12'] = df['x12']
    predict_data['x62'] = df['x62']
    predict_data['x58'] = df['x58']
    predict_data['x56'] = df['x56']

    predict_data["x5_saturday"] = np.where(df["x5"] == "saturday", 1, 0)
    predict_data["x5_monday"] = np.where(df["x5"] == "monday", 1, 0)
    predict_data["x81_December"] = np.where(df["x81"] == "December", 1, 0)
    predict_data["x81_July"] = np.where(df["x81"] == "July", 1, 0)
    predict_data["x31_japan"] = np.where(df["x31"] == "japan", 1, 0)
    predict_data["x81_October"] = np.where(df["x81"] == "October", 1, 0)
    predict_data["x5_sunday"] = np.where(df["x5"] == "sunday", 1, 0)
    predict_data["x31_asia"] = np.where(df["x31"] == "asia", 1, 0)
    predict_data["x81_February"] = np.where(df["x81"] == "February", 1, 0)

    predict_data["x81_May"] = np.where(df["x81"] == "May", 1, 0)
    predict_data["x5_monday"] = np.where(df["x5"] == "monday", 1, 0)
    predict_data["x81_September"] = np.where(df["x81"] == "September", 1, 0)
    predict_data["x81_March"] = np.where(df["x81"] == "March", 1, 0)

    predict_data["x81_November"] = np.where(df["x81"] == "November", 1, 0)

    predict_data["x81_June"] = np.where(df["x81"] == "June", 1, 0)

    predict_data["x5_tuesday"] = np.where(df["x5"] == "tuesday", 1, 0)
    predict_data["x81_August"] = np.where(df["x81"] == "August", 1, 0)
    predict_data["x81_January"] = np.where(df["x81"] == "January", 1, 0)

    predict_data["x31_germany"] = np.where(df["x31"] == "germany", 1, 0)


    prediction = model.predict(predict_data)
    outcome = pd.DataFrame(prediction).rename(columns={0: 'probs'})

    outcome['Business_Outcome'] = ["Event" if x >= 0.75 else "Not Event" for x in outcome['probs']]


    all_predictions = outcome.copy()

    all_predictions.rename(columns={'probs': 'phat'}, inplace=True)

    all_predictions['model_input'] = input_json[0].astype('string')
    all_predictions['model_input'] = all_predictions['model_input'].astype('string')




    return str(all_predictions.to_json(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1313)
