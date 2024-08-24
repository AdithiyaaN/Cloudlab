from flask import Flask, request, render_template
import numpy as np
from sklearn import linear_model
from sklearn import model_selection

app = Flask(__name__)

# Sample data
X = [[4.0],[5.0],[6.0],[7.0],[8.0],[9.0],[10.0],[11.0]]
y = [8, 10, 12, 14, 16, 18, 20, 22]

# Train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)

# Train the model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input feature from the form
        input_feature = float(request.form['feature'])
        prediction = reg.predict([[input_feature]])

        return render_template('index.html', prediction_text=f'Predicted Value: {prediction[0]:.2f}')
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter a numeric value.')

if __name__ == "__main__":
    app.run(debug=True)
