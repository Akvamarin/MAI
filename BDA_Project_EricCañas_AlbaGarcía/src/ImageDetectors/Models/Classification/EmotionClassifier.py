from ImageDetectors.Models.Constants import EMOTION_CLASSIFIER_DIR, MIN_CONFIDENCE
from os.path import join
from skimage.transform import resize
from skimage.color import rgb2gray
from keras.models import load_model

# This model was trained by: https://github.com/BishalLakha/Facial-Expression-Recognition-with-Keras
# It should be downloaded from his repository (and please star him if you do).
MODEL_PATH = join(EMOTION_CLASSIFIER_DIR, 'emotion_recognition.h5')

# Positions
CONFIDENCE = 2
X2, Y2 = -2, -1

EMOTIONS = ['Angry','Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionClassifier:
    """
    Given a face image in format uint8 RGB, process it through the emotion recognition model and predict its emotion
    """
    def __init__(self):
        """
        Implementation of the Emotion Recognition model offered under Free use license by:
        https://github.com/BishalLakha/Facial-Expression-Recognition-with-Keras
        """
        self.network = load_model(MODEL_PATH)
        self.input_size = self.network.input.shape[1:-1]

    def predict(self, face):
        """
        Predicts which is the label that would define the emotion of the given face. If its confidence is below of
        a confidence threshold, return Missing Value (None)
        :param face: Numpy image in format HxWx3 uint8. Face from which predict the emotion
        :return:
        """
        # Resize the face to the model input size
        face = resize(image=rgb2gray(face),output_shape=self.input_size)
        # Predict the probabilities of each emotion
        probabilities = self.network.predict(face[None,...,None])[0]
        # Take the most probable emotion
        max_prob = probabilities.argmax()
        # Take this label if the confidence is high enough, or Missing Value (None) elsewhere.
        emotion = EMOTIONS[max_prob] if probabilities[max_prob] > MIN_CONFIDENCE else None
        return emotion