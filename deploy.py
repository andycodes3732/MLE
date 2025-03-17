from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('house_price_predictor.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'predicted_price': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


{
    "features": [1.5, 5.5, 2.0, 1.5, 0.3, 1.0, 3.5, 2.0, 1.0, 4.0]
}


'''
# Use Python 3.8 image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose the port for the app
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]


Flask
joblib
numpy
scikit-learn

'''