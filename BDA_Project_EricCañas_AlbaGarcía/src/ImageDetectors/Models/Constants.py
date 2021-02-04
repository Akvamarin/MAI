import os
import numpy as np
from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12
from ImageDetectors.Models.Segmentation.FaceSegmentation.BiSeNetFaceParsing import bisenet_face_parsing

# Model Directories
DETECTION_MODELS_DIR = os.path.join('ImageDetectors', 'Models', 'Detection')
CLASSIFICATION_MODELS_DIR = os.path.join('ImageDetectors', 'Models', 'Classification')
ESTIMATION_MODELS_DIR = os.path.join('ImageDetectors', 'Models', 'Estimation')
FACE_DETECTOR_DIR = os.path.join(DETECTION_MODELS_DIR, 'OpenCVFaceDetectorModel')
AGE_ESTIMATOR_DIR = os.path.join(ESTIMATION_MODELS_DIR, 'AgePredictor')
GENDER_ESTIMATOR_DIR = os.path.join(CLASSIFICATION_MODELS_DIR, 'GenderPredictor')
EMOTION_CLASSIFIER_DIR =  os.path.join(CLASSIFICATION_MODELS_DIR, 'EmotionClassifier')

# Segmentation Models Information
ADE20K, CITY_SCAPES, VOC12, FACE_PARSING = 'ADE20K', 'CITY_SCAPES', 'VOC12', 'FACE_PARSING'

# CITY_SCAPES should not be used at least it is sure that the image is from a road. Best model is ADE20K
MODEL_BY_DATASET = {ADE20K : pspnet_50_ADE_20K, CITY_SCAPES : pspnet_101_cityscapes, VOC12 : pspnet_101_voc12,
                    FACE_PARSING : bisenet_face_parsing}
MEANS_BY_DATASET = {ADE20K : None, CITY_SCAPES : None, VOC12: None, FACE_PARSING : np.array((0.485, 0.456, 0.406))}
STD_BY_DATASET = {ADE20K : None, CITY_SCAPES : None, VOC12 : None, FACE_PARSING : np.array((0.229, 0.224, 0.225))}

# Dataset CSV Directories and information
BASE_DATASET_DESCRIPTION_PATH = os.path.join('ImageDetectors', 'Models', 'Segmentation', 'ClassInfo')
VALID_DATASETS = {os.path.splitext(f)[0] for f in os.listdir(BASE_DATASET_DESCRIPTION_PATH)}
MULTIPLE_CLASSES_SEPARATOR = ';'

# Output parameters
MIN_CONFIDENCE = 0.5
DEFAULT_MARGIN_AROUND_FACE = 0.2
DECIMALS = 2
