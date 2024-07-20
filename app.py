from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# #mood tracker
# model = joblib.load('model.pkl')

# # Define mood labels corresponding to numerical predictions
# mood_labels = ['happy', 'sad', 'anxious', 'angry', 'neutral']

# @app.route('/predicts',methods=['POST'])
# def predict():
#     # Get user input from the form
#     activity_level = float(request.form['activity_level'])
#     sleep_quality = float(request.form['sleep_quality'])

#     # Example of predicting mood based on input (replace with your actual prediction logic)
#     # Assuming your model expects an array of features
#     input_data = np.array([[activity_level, sleep_quality]])
#     predicted_mood_index = model.predict(input_data)[0]  # Get the predicted mood index

#     # Map the predicted index to the corresponding mood label
#     predicted_mood = mood_labels[predicted_mood_index]

#     # Render the predict.html template with prediction results
#     return render_template('result.html', 
#                            activity_level=activity_level, 
#                            sleep_quality=sleep_quality, 
#                            predicted_mood=predicted_mood)



#1Objective

from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Define file paths
model_file = 'workflow_model1.pkl'
scaler_file = 'scaler1.pkl'
le_file = 'label_encoder1.pkl'

# Initialize global variables
model = None
scaler = None
le = None

@app.route('/train1', methods=['POST'])
def train():
    global model, scaler, le

    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    # Check if the file is a CSV
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    try:
        # Read the uploaded file
        df = pd.read_csv(file)

        # Check if the DataFrame contains the necessary columns
        required_columns = ['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                            'Staff_Availability_Rate', 'Equipment_Utilization', 'Workflow_Status']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'CSV must contain required columns'}), 400
        
        # Encode the target variable
        le = LabelEncoder()
        df['Workflow_Status'] = le.fit_transform(df['Workflow_Status'])

        # Prepare features and target
        X = df[['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                'Staff_Availability_Rate', 'Equipment_Utilization']]
        y = df['Workflow_Status']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

        # Initialize and fit the scaler
        scaler = StandardScaler()
        X_train[['Patient_Count', 'Average_Wait_Time']] = scaler.fit_transform(X_train[['Patient_Count', 'Average_Wait_Time']])
        X_test[['Patient_Count', 'Average_Wait_Time']] = scaler.transform(X_test[['Patient_Count', 'Average_Wait_Time']])

        # Initialize and train the Random Forest classifier
        model = RandomForestClassifier(random_state=2529)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Save the model and the scaler
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model, label encoder, and scaler
        joblib.dump(model, 'workflow_model1.pkl')
        joblib.dump(le, 'label_encoder1.pkl')
        joblib.dump(scaler, 'scaler1.pkl')

        return jsonify({"message": accuracy*100})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict1', methods=['POST'])
def predict():
    global model, scaler, le
    
    try:
        if model is None or scaler is None or le is None:
            return jsonify({'error': 'Model not trained or files not found'}), 500
        
        # Get JSON data from the request
        data = request.json
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess data
        df[['Patient_Count', 'Average_Wait_Time']] = scaler.transform(df[['Patient_Count', 'Average_Wait_Time']])
        
        # Make predictions
        predictions = model.predict(df)
        predicted_status = le.inverse_transform(predictions)
        
        # Return JSON response
        return jsonify({'workflow_status': predicted_status[0]})
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 500

@app.route('/1')
def home1():
    return render_template("index1.html")



#objective4




#5objective
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model

# Load the trained model
model = load_model('pneumonia_detection_model5.h5')

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define a function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Define the prediction route
@app.route('/predict5', methods=['POST'])
def predict5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_path = f'uploads/{file.filename}'
    file.save(img_path)

    img = preprocess_image(img_path)
    prediction = model.predict(img)

    result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    confidence = float(prediction[0][0])

    return jsonify({'prediction': result, 'confidence': confidence})

@app.route("/5")
def home5():
    return render_template("index5.html")


#objective6

# Load the trained model
model = joblib.load('trained_model6.pkl')

# LabelEncoder for categorical variables
label_encoders = {}

def encode_categorical(data):
    global label_encoders
    
    # Define categorical columns
    categorical_cols = ['Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2', 'Medication1', 'Medication2']
    
    # Initialize or update LabelEncoders
    for col in categorical_cols:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(data[col].astype(str))  # Ensure consistent type
    
    # Encode categorical variables
    for col in categorical_cols:
        # Handle unseen categories by assigning a default value or NaN
        data[col] = data[col].astype(str).map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else None)
    
    return data

@app.route('/6')
def home6():
    # Render the HTML page
    return render_template('index6.html')

@app.route('/predict6', methods=['POST'])
def predict6():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical variables
        df = encode_categorical(df)

        # Ensure all columns expected by the model are present
        expected_cols = ['Age', 'Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2',
                         'BP_Systolic', 'BP_Diastolic', 'Glucose', 'BMI', 'Medication1', 'Medication2',
                         'HospitalAdmissions']

        for col in expected_cols:
            if col not in df.columns:
                df[col] = None  # Assign None or NaN for missing columns

        # Reorder columns to match the order of training data
        df = df[expected_cols]

        # Make prediction
        prediction = model.predict(df)

        # Assuming 'prediction' is a numpy array or similar, convert to string or JSON format
        outcome_mapping = {
            0: 'Died',
            1: 'Stable',
            2: 'Improved',
            3: 'Survived'  # Assuming prediction[0] can go up to 3 based on your previous mapping
        }

        # Adjust the prediction_result based on the mapping
        prediction_result = {
            'prediction': outcome_mapping[prediction[0]]
        }
        # prediction_result = {
        #     'prediction': str(prediction[0])  # Adjust this based on your prediction format
        # }

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400



#Objective 7



#objective moode tracker using image
model = load_model('model/model9.h5')

# Dictionary to map numerical labels to emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values

# Route to index page
@app.route('/9')
def home9():
    return render_template('index9.html', prediction_result=None)

# Route to handle prediction
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction_result='No file part')
        
        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction_result='No selected file')

        if file:
            # Save the file to the uploads folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            predicted_emotion = emotion_labels[predicted_label]

            # Delete the uploaded file to save space
            os.remove(file_path)

            return render_template('index9.html', prediction_result=predicted_emotion)

# @app.route("/7")
# def home7():
#     return render_template("index7.html")



#10 objective
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocess text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into a single string
    return ' '.join(tokens)

# Load and preprocess the dataset
data = pd.read_csv('data_disease.csv')
data['processed_text'] = data['Symptoms'].apply(preprocess_text)

# Train the model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['processed_text'])
y = data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

@app.route('/10')
def home10():
    return render_template('index10.html')

@app.route('/output', methods=['POST'])
def output():
    symptoms = request.json.get('symptoms', '')
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400

    symptoms = preprocess_text(symptoms)
    vectorized_input = tfidf.transform([symptoms])
    prediction = model.predict(vectorized_input)[0]

    # Debugging output
    print(f"Predicted Disease: {prediction}")

    disease_info = data[data['Disease'] == prediction]
    if not disease_info.empty:
        disease_info = disease_info.iloc[0]
        description = disease_info['Description']
        solution = disease_info['Solution']
    else:
        description = 'Description not available'
        solution = 'Solution not available'

    # Debugging output
    print(f"Description: {description}")
    print(f"Solution: {solution}")

    return jsonify({'disease': prediction, 'description': description, 'solution': solution})






import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
# #4Objective
# # Load the trained model
with open('remote_patient_monitoring_model4.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/4')
def home4():
    return render_template('index4.html')

@app.route('/resulti', methods=['POST'])
def resulti():
    # Extract input data from form
    heart_rate = float(request.form['heart_rate'])
    blood_pressure = float(request.form['blood_pressure'])
    temperature = float(request.form['temperature'])
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[heart_rate, blood_pressure, temperature]],columns=['heart_rate', 'blood_pressure', 'temperature'])
    
    # Make predictions
    prediction = model.predict(input_data)
    probability_estimates = model.predict_proba(input_data)
    
    # Determine risk status
    risk_status = 'At Risk' if probability_estimates[0][1] > 0.5 else 'Not at Risk'
    risk_value = probability_estimates[0][1] if probability_estimates[0][1] > 0.5 else probability_estimates[0][0]
    
    # Format the result
    result = {
        'prediction': int(prediction[0]),
        'risk_status': risk_status,
        'risk_value': f"{risk_value * 100:.2f}%",
        'probability': {
            'not_at_risk': f"{probability_estimates[0][0] * 100:.2f}%",
            'at_risk': f"{probability_estimates[0][1] * 100:.2f}%"
        }
    }
    
    return render_template('predict4.html',  heart_rate=heart_rate,blood_pressure=blood_pressure,temperature=temperature,
                           result=result)




#11objective diabetes
with open('diabetes.pkl', 'rb') as f:
    clf = pickle.load(f)

@app.route('/outcome', methods=['POST'])
def outcome():
    data = request.get_json()
    pregnancies = data['pregnancies']
    glucose = data['glucose']
    diastolic = data['diastolic']
    triceps = data['triceps']
    insulin = data['insulin']
    bmi = data['bmi']
    dpf = data['dpf']
    age = data['age']

    # Create a new data point
    new_data = [[pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age]]

    # Make a prediction
    prediction = clf.predict(new_data)[0]

    # Return the prediction
    return jsonify({'diabetes': int(prediction)})

@app.route("/11")
def diabetes():
    return render_template("index11.html")
if __name__=="__main__":
    app.run(debug=True,port=5000)