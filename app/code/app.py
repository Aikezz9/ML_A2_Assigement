# # # from flask import Flask, request, jsonify, render_template
# # # import pickle
# # # import numpy as np
# # # from model.linear_regression_class import *

# # # # Initialize Flask app
# # # app = Flask(__name__)

# # # # Load the model_1
# # # with open('/Users/issac/ML/A2_cars_price_prediction/app/code/model/car_price_prediction_a1.model', 'rb') as file:
# # #     model = pickle.load(file)

# # # # Define the home route
# # # @app.route('/')
# # # def home():
# # #     return render_template('index.html')  # HTML frontend

# # # # Define the prediction route
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Extract data from the request
# # #         data = request.json  # Expecting JSON input
# # #         max_power = float(data['max_power'])
# # #         mileage = float(data['mileage'])
# # #         engine = float(data['engine'])
        
# # #         # Prepare input for prediction
# # #         input_features = [[max_power, mileage, engine]]
# # #         prediction = np.exp(model.predict(input_features)[0])

# # #         # Return prediction
# # #         return jsonify({'prediction': prediction})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)})


# # # # Load the model_2
# # # with open('/Users/issac/ML/A2_cars_price_prediction/app/code/model/car_price_prediction_a1.model', 'rb') as file:
# # #     model = pickle.load(file)

# # # # Define the home route
# # # @app.route('/index_a2')
# # # def home():
# # #     return render_template('index_a2.html')  # HTML frontend

# # # # Define the prediction route
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Extract data from the request
# # #         data = request.json  # Expecting JSON input
# # #         max_power = float(data['max_power'])
# # #         mileage = float(data['mileage'])
# # #         year = float(data['year'])
        
# # #         # Prepare input for prediction
# # #         input_features = [[max_power, mileage, year]]
# # #         prediction = np.exp(model.predict(input_features)[0])

# # #         # Return prediction
# # #         return jsonify({'prediction': prediction})
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)})

# # # if __name__ == '__main__':
# # #     app.run(host='0.0.0.0', port=5000, debug=True)


# # #Testing 2
# # from flask import Flask, request, jsonify, render_template
# # import pickle
# # import numpy as np
# # from model.linear_regression_class import *

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load Model 1 for index.html
# # with open('/Users/issac/ML/A2_cars_price_prediction/app/code/model/car_price_prediction_a1.model', 'rb') as file:
# #     model_1 = pickle.load(file)

# # # Load Model 2 for index_a2.html
# # with open('/Users/issac/ML/A2_cars_price_prediction/app/code/model/car_price_prediction_a2.model', 'rb') as file:
# #     model_2 = pickle.load(file)

# # # Home Page (First Page)
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # # Second Page
# # @app.route('/index_a2')
# # def second_page():
# #     return render_template('index_a2.html')

# # # Prediction for First Page (Model 1)
# # @app.route('/predict_a1', methods=['POST'])
# # def predict_a1():
# #     try:
# #         data = request.json  # Expecting JSON input
# #         max_power = float(data['max_power'])
# #         mileage = float(data['mileage'])
# #         engine = float(data['engine'])

# #         # Prepare input for prediction
# #         input_features = [[max_power, mileage, engine]]
# #         prediction = np.exp(model_1.predict(input_features)[0])

# #         return jsonify({'prediction': prediction})
# #     except Exception as e:
# #         return jsonify({'error': str(e)})

# # # Prediction for Second Page (Model 2)
# # @app.route('/predict_a2', methods=['POST'])
# # def predict_a2():
# #     try:
# #         data = request.json  # Expecting JSON input
# #         max_power = float(data['max_power'])
# #         mileage = float(data['mileage'])
# #         year = float(data['year'])

# #         # Prepare input for prediction
# #         input_features = [[max_power, mileage, year]]
# #         prediction = np.exp(model_2.predict(input_features)[0])

# #         return jsonify({'prediction': prediction})
# #     except Exception as e:
# #         return jsonify({'error': str(e)})

# # if __name__ == '__main__':
# #     app.run(host='0.0.0.0', port=5000, debug=True)


# from flask import Flask, request, jsonify, render_template
# import pickle
# import os
# import numpy as np
# from model.linear_regression_class import *

# app = Flask(__name__)

# # Load Model 1
# with open('./model/car_price_prediction_a1.model', 'rb') as file:
#     model_1 = pickle.load(file)

# # Load Model 2
# with open('./model/car_price_prediction_a2.model', 'rb') as file:
#     model_2 = pickle.load(file)


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/index_a2')
# def second_page():
#     return render_template('index_a2.html')

# @app.route('/compare')
# def compare():
#     return render_template('compare.html')

# @app.route('/predict_a1', methods=['POST'])
# def predict_a1():
#     try:
#         data = request.json
#         max_power = float(data['max_power'])
#         mileage = float(data['mileage'])
#         engine = float(data['engine'])

#         input_features = [[max_power, mileage, engine]]
#         prediction = np.exp(model_1.predict(input_features)[0])

#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# @app.route('/predict_a2', methods=['POST'])
# def predict_a2():
#     try:
#         data = request.json
#         max_power = float(data['max_power'])
#         mileage = float(data['mileage'])
#         year = float(data['year'])

#         input_features = [[max_power, mileage, year]]
#         prediction = np.exp(model_2.predict(input_features)[0])

#         return jsonify({'prediction': prediction})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from linear_regression_class import *
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# ✅ Tell Flask to look for templates in "template/" instead of "templates/"
app = Flask(__name__, template_folder=("template"))

# Get the absolute path to the "model" directory
model_dir = os.path.join(os.path.dirname(__file__), "model")

# Load Model 1 (Old Model)
model_1_path = os.path.join(model_dir, "car_price_prediction_a1.model")
with open(model_1_path, 'rb') as file:
    model_1 = pickle.load(file)

# Load Model 2 (New Model)
model_2_path = os.path.join(model_dir, "car_price_prediction_a2.model")
with open(model_2_path, 'rb') as file:
    model_2 = pickle.load(file)


# 🏠 **Main Home Page - Choose a Model**
@app.route('/')
def home():
    return render_template('home.html')  # ✅ Flask will now correctly find home.html

# 📌 **Old Model Page**
@app.route('/index_a1')
def old_model():
    return render_template('index_a1.html')

# 🌟 **New Model Page**
@app.route('/index_a2')
def new_model():
    # ✅ Define default values to pass to the template
    default_values = {
        'max_power': '',
        'mileage': '',
        'year': ''
    }
    return render_template('index_a2.html', default_values=default_values)  # ✅ Pass default_values


# 🔄 **Comparison Page**
@app.route('/compare')
def compare():
    return render_template('compare.html')

# 📈 **Prediction for Old Model (Model 1)**
@app.route('/predict_a1', methods=['POST'])
def predict_a1():
    try:
        data = request.json
        max_power = float(data['max_power'])
        mileage = float(data['mileage'])
        engine = float(data['engine'])

        input_features = [[max_power, mileage, engine]]
        prediction = np.exp(model_1.predict(input_features)[0])

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# 📊 **Prediction for New Model (Model 2)**
@app.route('/predict_a2', methods=['POST'])
def predict_a2():
    try:
        data = request.json
        max_power = float(data['max_power'])
        mileage = float(data['mileage'])
        year = float(data['year'])

        input_features = [[max_power, mileage, year]]
        prediction = np.exp(model_2.predict(input_features)[0])

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030, debug=True)

