
import numpy as np
import cv2,os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
from deepface import DeepFace
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, send_from_directory

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app=Flask(__name__)


# Load the dataset
df = pd.read_csv('data_stress.csv')  # Replace with your dataset file path

df.drop(['blood oxygen ','limb movement','eye movement'], axis=1,inplace=True)

# Define the SimpleImputer to handle missing values for the entire dataset
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# Apply the imputer to the entire dataset (excluding 'Stress Levels')
df_imputed = pd.DataFrame(imp.fit_transform(df.drop('Stress Levels', axis=1)))

# Replace the original column names after imputation
df_imputed.columns = df.drop('Stress Levels', axis=1).columns

# Add the target column back to the dataframe
df_imputed['Stress Levels'] = df['Stress Levels']

# Define the pipeline for KNN model
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('KNN', KNeighborsClassifier(n_neighbors=4))
])

# Split the dataset
X = df_imputed.drop('Stress Levels', axis=1)
y = df_imputed['Stress Levels']

# Fit the pipeline to the data
pipe.fit(X, y)

# Predict on the training data
y_pred = pipe.predict(X)

# Calculate accuracy score on the training data
accuracy = accuracy_score(y, y_pred)

accuracy = round(accuracy,2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['POST','GET'])
def contact():
    if request.method == 'POST':
        # Get the input values from the form (ensure you get the right input names from your HTML)
        body_temperature = float(request.form['body_temperature'])
        hours_of_sleep = float(request.form['hours_of_sleep'])
        heart_rate = float(request.form['heart_rate'])
        snoring_range = float(request.form['snoring_range'])
        respiration_rate = float(request.form['respiration_rate'])
        
        # Prepare the input data for prediction
        input_data = np.array([[body_temperature, hours_of_sleep, heart_rate, snoring_range, respiration_rate]])  # Adjust this as necessary
        
        # Scale the input data and predict the stress level
        scaled_input = pipe.named_steps['scaler'].transform(input_data)
        predicted_stress_level = pipe.named_steps['KNN'].predict(scaled_input)
        
        # Map the predicted stress level to a corresponding message
        stress_level = predicted_stress_level[0]
        
        if stress_level == 0:
            stress_message = "Your stress level is very low. You're feeling calm and relaxed!"
        elif stress_level == 1:
            stress_message = "Your stress level is low. You may have some mild stress, but it's manageable."
        elif stress_level == 2:
            stress_message = "Your stress level is moderate. It's important to take a break and relax."
        elif stress_level == 3:
            stress_message = "Your stress level is high. Take some time to relax and consider stress-relieving activities."
        elif stress_level == 4:
            stress_message = "Your stress level is very high. It's important to seek support and focus on stress management strategies."

        return render_template("stress.html", prediction=stress_message, stress_level = stress_level,  accuracy=accuracy, respiration_rate = respiration_rate, snoring_range = snoring_range, heart_rate = heart_rate, hours_of_sleep = hours_of_sleep,  body_temperature = body_temperature, )

    return render_template('contact.html')

@app.route('/img_classification',methods=["POST","GET"])
def img_classification():
    if request.method=='POST':
        myfile=request.files['file']
        fn=myfile.filename
        mypath=os.path.join('images/', fn)
        myfile.save(mypath)
        accepted_formated=['jpg','png','jpeg','jfif']
        if fn.split('.')[-1] not in accepted_formated:
            msg = "Image formats only Accepted"      
        else:
            model = load_model(r'models\FinalModel.h5')
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = test_image/255
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            val =  np.argmax(result, axis=1)[0]
            print(result)
            result = round(float(np.max(result, axis=1)[0]),2)
            
            classes=['Angry Based Stress','Disgust Based Stress','Fear Based Stress',
                'Happy - No Stress','Neutral - No Stress','Sad Based Stress',
                'Surprise - No Stress']
            prediction=classes[val]
            print(prediction)
        return render_template("result.html",image_name=fn, text=prediction, score = result)

    return render_template('result.html')

@app.route('/img_classification/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/live')
def live():
    
    model = load_model(r'models\emotion_detection.model')

    # open webcam
    webcam = cv2.VideoCapture(0)


    # loop through frames
    while webcam.isOpened():
        # read frame from webcam
        status, frame = webcam.read()

        if not status:
            print("Failed to grab frame")
            break

        # apply face detection
        face, confidence = cv.detect_face(frame)

        # Analyze the frame for emotion and age detection
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Extract the dominant emotion and age

        dominant_emotion = result[0]['dominant_emotion']
        # predicted_age = result[0]['age']
        if dominant_emotion == "angry" or "disgust" or "fear" or "sad": 
            dominant_emotion = "Stress"
        else:
            dominant_emotion = "No Stress"
        # initialize a flag to check if any face is processed
        face_detected = False

        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0] < 10) or (face_crop.shape[1] < 10):
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix

            # get label with max accuracy
            idx = np.argmax(conf)
            # label = classes[idx]

            label = "{}{:.2f}%".format("",conf[idx] * 100)

            # Mark that a face has been detected
            face_detected = True

        # Display emotion and age on the frame if face is detected
        if face_detected:
            cv2.putText(frame, f"Emotion - {dominant_emotion} {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"Age: {predicted_age}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # display output
        cv2.imshow("Detection", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release resources after the loop ends
    webcam.release()
    cv2.destroyAllWindows()
    return render_template('index.html')


if __name__=='__main__':
    app.run(debug=True)