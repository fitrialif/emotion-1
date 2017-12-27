import cv2
import sys
import json
import time
import numpy as np
from keras.models import model_from_json


def load_model(architecture_fname, weights_fname):
    json_file = open(architecture_fname,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_fname)
    return model

def predict_emotion(classifier, face_image_gray):
    """
    use a trained convolutional network to predict emotion from grayscale images. 

    returns
    -------
    emotions : list
        probability distribution for emotions.
    """
    resized_img = cv2.resize(face_image_gray, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    list_of_list = classifier.predict(image, batch_size=1, verbose=1)
    angry, fear, happy, sad, surprise, neutral = [prob for lst in list_of_list for prob in lst]
    emotions = [angry, fear, happy, sad, surprise, neutral]
    return emotions


if __name__ == "__main__":

    # load haar cascade face detector
    cascPath = sys.argv[1]
    faceCascade = cv2.CascadeClassifier(cascPath)

    # load emotion classifier
    classifier = load_model("model.json", "model.h5")

    # init webcam
    video_capture = cv2.VideoCapture(0)  # 0 = webcam

    while True:
        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY,1)

        # fetch faces
        faces = faceCascade.detectMultiScale(img_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30), 
            flags=cv2.CASCADE_SCALE_IMAGE)

        # for each face: draw bounding box, predict emotions, and display predictions
        for (x, y, w, h) in faces:

            face_image_gray = img_gray[y:y+h, x:x+w]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            angry, fear, happy, sad, surprise, neutral = predict_emotion(classifier, face_image_gray)
            
            # display classifications
            cv2.putText(frame, "angry " + str(angry), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(frame, "scared " + str(fear), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(frame, "happy " + str(happy), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(frame, "sad " + str(sad), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(frame, "surprise " + str(surprise), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
            cv2.putText(frame, "neutral " + str(neutral), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
