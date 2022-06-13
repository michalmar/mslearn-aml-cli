import joblib
import numpy as np
import os
import json

# from inference_schema.schema_decorators import input_schema, output_schema
# from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

# The init() method is called once, when the web service starts up.
#
# Typically you would deserialize the model file, as shown here using joblib,
# and store it in a global variable so your run() method can access it later.


def init():
    global model
    global local_debug
    global init_error_dict
    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    # model_filename = 'diabetes-sklearn-manual'
    local_debug = True
    model_loaded = False

    if (not model_loaded):
        try:
            # attempt 1 / remote / no model name specified
            model_filename = None
            model_path = os.environ['AZUREML_MODEL_DIR']
            model = joblib.load(model_path)
            init_error_dict = {
                "status":
                    "Succeeded",
                "message":
                    f"attempt 1 / remote / no model name specified, loaded correctly from path: {model_path}"
            }
            model_loaded = True
        except FileNotFoundError:
            init_error_dict = {
                "status":
                    "Failed",
                "message":
                    f"attempt 1 / remote / no model name specified, Model path: {model_path} not found"
            }
            print(f'Message: {init_error_dict["message"]}')
            model = None
            model_loaded = False
            pass
        except IsADirectoryError:
            init_error_dict = {
                "status":
                    "Failed",
                "message":
                    f"attempt 1 / remote / no model name specified, Model path: {model_path} IsADirectoryError"
            }
            print(f'Message: {init_error_dict["message"]}')
            # Driver function

            for (root, dirs, files) in os.walk(model_path, topdown=True):
                print(f"root: {root}")
                print(f"dirs: {dirs}")
                print(f"files: {files}")
                print('--------------------------------')

            model = None
            model_loaded = False
            pass

    if (not model_loaded):
        try:
            # attempt 2 / remote / model name specified
            model_filename = 'model.pkl'
            model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'],
                                      model_filename)
            model = joblib.load(model_path)
            init_error_dict = {
                "status":
                    "Succeeded",
                "message":
                    f"attempt 2 / remote / model name specified, loaded correctly from path: {model_path}"
            }
            model_loaded = True
        except FileNotFoundError:
            init_error_dict = {
                "status":
                    "Failed",
                "message":
                    f"attempt 2 / remote / model name specified, Model path: {model_path} not found"
            }
            print(f'Message: {init_error_dict["message"]}')
            model = None
            model_loaded = False
            pass

    if (not model_loaded):
        try:
            # attempt 3 / locally / model name = local file
            model_filename = 'model.pkl'  # locally
            model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'],
                                      "model_man_deploy", model_filename)
            model = joblib.load(model_path)
            init_error_dict = {
                "status":
                    "Succeeded",
                "message":
                    f"attempt 3 / locally / model name = local file, loaded correctly from path: {model_path}"
            }
            model_loaded = True
        except FileNotFoundError:
            init_error_dict = {
                "status":
                    "Failed",
                "message":
                    f"attempt 3 / locally / model name = local file, Model path: {model_path} not found"
            }
            print(f'Message: {init_error_dict["message"]}')
            model = None
            model_loaded = False
            pass

    print("\n\n")
    print("*" * 60)
    print("Init function end, result:")
    print(f'Status:  {init_error_dict["status"]}')
    print(f'Message: {init_error_dict["message"]}')
    print(f'Model:   {model}')


# The run() method is called each time a request is made to the scoring API.
#
# Shown here are the optional input_schema and output_schema decorators
# from the inference-schema pip package. Using these decorators on your
# run() method parses and validates the incoming payload against
# the example input you provide here. This will also generate a Swagger
# API document for your web service.
# @input_schema('data', NumpyParameterType(np.array([[2,180,74,24,21,23.9091702,1.488172308,60]])))
# @output_schema(NumpyParameterType(np.array([1.0])))
# def run(data):
#     print("\n\n")
#     print("*" * 60)
#     print("this is data:")
#     # print(data)
#     data_json = json.loads(data)
#     print(data_json["input_data"]["data"])
#     print("*" * 60)
#     print(init_error_dict["message"])
#     # Use the model object loaded by init().
#     result = model.predict(data_json["input_data"]["data"])


#     # You can return any JSON-serializable object.
#     return result.tolist()
def run(mini_batch):
    print(f"run method start: {__file__}, run({mini_batch})")
    resultList = []

    for f in mini_batch:
        # prepare each image
        # TODO
        print(f"processing file {f}")
        # resultList.append("{}: {}".format(os.path.basename(image), best_result))

    return resultList


if __name__ == '__main__':
    init()
    run(["./Allfiles/Labs/04/mlflow-endpoint/data/diabetes_classification.csv"])