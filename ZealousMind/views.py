import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.shortcuts import render
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn import svm
import joblib

# Load the saved model using the absolute path
mine_modal_path = os.path.join(settings.BASE_DIR, 'models', 'logistic_model.pkl')
mine_model = joblib.load(mine_modal_path)


# Load the saved model components
tokenizer_path = os.path.join(settings.STATICFILES_DIRS[0], 'textclassifier', 'tokenizer.pickle')
label_encoder_path = os.path.join(settings.STATICFILES_DIRS[0], 'textclassifier', 'label_encoder.pickle')
model_path = os.path.join(settings.STATICFILES_DIRS[0], 'textclassifier', 'text_model.keras')

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(label_encoder_path, 'rb') as handle:
    label_encoder = pickle.load(handle)

model = load_model(model_path)

def home(request):
    return render(request, 'home.html')

def diabetes(request):
    return render(request, 'diabetes.html')

def result(request):
    # Construct the path to the diabetes CSV file
    file_path = os.path.join(settings.STATICFILES_DIRS[0], 'diabetesPridiction', 'data', 'diabetes.csv')
    
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Prepare features and label
    X = data.drop(columns='Outcome', axis=1)
    Y = data['Outcome']
    
    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Initialize and train the SVM classifier (consider saving and loading a pre-trained model instead)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    
    # Retrieve input values
    val1 = float(request.GET.get('n1', 0))
    val2 = float(request.GET.get('n2', 0))
    val3 = float(request.GET.get('n3', 0))
    val4 = float(request.GET.get('n4', 0))
    val5 = float(request.GET.get('n5', 0))
    val6 = float(request.GET.get('n6', 0))
    val7 = float(request.GET.get('n7', 0))
    val8 = float(request.GET.get('n8', 0))
    
    # Make prediction
    prediction = classifier.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    
    # Result based on prediction
    result2 = "You are Diabetic" if prediction == 1 else "You are Not Diabetic"
    
    return render(request, 'diabetes.html', {'result': result2})

def mine(request):
    result = None

    if request.method == 'POST':
        # Get the input data from the textarea and process it
        input_data = request.POST.get('input_data', '').strip()  # Get input data and strip whitespace

        if input_data:
            try:
                # Convert the comma-separated input string into a list of floats
                input_data_list = list(map(float, input_data.split(',')))

                # Ensure the input data is reshaped correctly for prediction (1 sample, 60 features)
                input_data_np = np.asarray(input_data_list).reshape(1, -1)

                # Make prediction
                prediction = mine_model.predict(input_data_np)

                # Check if it's Rock or Mine
                if prediction[0] == 'R':
                    result = "The object is Rock"
                else:
                    result = "The object is Mine"
            except ValueError as ve:
                print(f"ValueError: {ve}")
                result = "Please enter valid numbers separated by commas."
            except Exception as e:
                print(f"Error: {e}")
                result = "An error occurred during prediction."
        else:
            result = "Please provide input data."

    return render(request, 'mine.html', {'result': result})

def textclassifier(request):
    result = None
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        if input_text:
            try:
                input_sequence = tokenizer.texts_to_sequences([input_text])
                max_length = model.input_shape[1]
                padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)

                prediction = model.predict(padded_input_sequence)
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])

                result = predicted_label[0]
            except Exception as e:
                print(f"Error: {e}")
                result = "An error occurred during classification."

    return render(request, 'textclassifier.html', {'result': result})


