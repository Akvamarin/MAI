import os
from ImageDetectors.Models.Segmentation.FaceSegmentation.PretrainedModel.BiSeNet import BiSeNet_keras
MODEL_PATH = os.path.join('ImageDetectors', 'Models', 'Segmentation', 'FaceSegmentation', 'PretrainedModel', 'BiSeNet_keras.h5')

def bisenet_face_parsing():
    """
    Returns the pretrained BiSeNet model for face parsing
    :return:
    """
    network = BiSeNet_keras()
    network.load_weights(MODEL_PATH)
    return network
