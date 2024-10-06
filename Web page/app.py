from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Load the trained model
model = pickle.load(open('Web page/model.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template("index.html")

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form input and convert to integers
    int_features = [int(x) for x in request.form.values()]
    
    # Ensure exactly 3 features are provided (Age, Gender, Family History)
    if len(int_features) != 3:
        return "Error: Expected 3 features (Age, Gender, Family History) but got {}".format(len(int_features))
    
    # Convert input to the format expected by the model
    final = [np.array(int_features)]
    
    # Print inputs for debugging
    print(f"Received input features: {int_features} (length: {len(int_features)})")
    print(f"Final input to the model: {final}")
    
    # Make prediction using the model
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    # Return prediction result based on the threshold
    if output > str(0.5):
        return render_template('index.html', pred='You need treatment. Probability of mental illness is {}'.format(output))
    else:
        return render_template('index.html', pred='You do not need treatment. Probability of mental illness is {}'.format(output))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
