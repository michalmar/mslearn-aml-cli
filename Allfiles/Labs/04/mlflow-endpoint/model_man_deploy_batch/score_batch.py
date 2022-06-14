import joblib
import numpy as np
import os
import json
import pandas as pd

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.


def init():
    global model
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_filename = 'model.pkl'
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'], "model",
                              model_filename)
    model = joblib.load(model_path)

    print("\n\n" + "*" * 60)
    print(f'Model:   {model}')


# The run() method is called each time a request is made to the scoring API.
def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []

    for f in mini_batch:
        print(f"processing file {f}")
        df = pd.read_csv(f)
        X = df[[
            'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'
        ]].values

        result = model.predict(X)

    return result.tolist()