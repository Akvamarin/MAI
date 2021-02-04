import torch
import Database.Constants as FIELDS
import numpy as np
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CLASS_POSITION = -1
H_POSITION, W_POSITION = 2, 3
AVAILABLE_MODELS_AT_HUB = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
BEST_MODEL = 'yolov5x'
TORCH_HUB_REPOSITORY = 'ultralytics/yolov5'


class YOLOv5:
    """
    Convenient implementation of the pretrained YOLOv5 model offered by 'ultralytics' through torch hub. It is used
    to detect the objects that appears in the image, its quantity and the approximated areas that the occupy.
    """
    def __init__(self, model_repository = TORCH_HUB_REPOSITORY, version=BEST_MODEL):
        # Assert that it is one of the available models of YOLOv5
        assert version in AVAILABLE_MODELS_AT_HUB
        # Download the model from the open source repository
        self.model = torch.hub.load(model_repository, model=version, pretrained=True).fuse().eval()
        # Preprocess the shape of inputs (PIL/cv2/numpy) and applies Non Max Supression.
        self.model = self.model.autoshape()
        # Take the class names
        self.names = self.model.names
        # Put the model into the selected device
        self.model.to(device=dev)

    def get_details(self, rgb_uint8_image):
        """
        Take a rgb uint8 image, and returns the objects that appear in it, the amount of each object
        and the approximated area that they ocuppy in the image.
        :param image_batch: WxHx3 numpy array containing the image to predict
        :return: Dictionary. Dictionary with the objects that appear in the image, the amount of each object
        and the approximated area that they ocuppy in the image.
        """
        # Predict the bounding boxes (xywhn refers to (X, Y, Weight, Height) Normalized)
        image_boxes = self.model(rgb_uint8_image).xywhn[0].detach().cpu().numpy()
        # For all the boxes (by default with a confidence higher than 0.25) in the image
        if len(image_boxes) > 0:
            objects = {}
            # Count the amount of objects of each type that does appear in the image
            classes_found = (image_boxes[...,CLASS_POSITION]).astype(np.int)
            # Calculate the area of every object
            objects_areas = image_boxes[...,W_POSITION]*image_boxes[...,H_POSITION]
            # Make groups with the objects of the same class
            object_ids, counts = np.unique(classes_found, return_counts=True)
            # Save the name of those grouped objects
            objects[FIELDS.OBJECT] = [self.names[object_id] for object_id in object_ids]
            # Save the quantity of those groups
            objects[FIELDS.COUNT] = [int(count) for count in counts]
            # Sum the area of the objects within the same group
            objects[FIELDS.TOTAL_AREA] = [float(np.sum(objects_areas[classes_found == object_id])) for object_id in object_ids]
        else:
            # If no objects were detected, set it as Missing Value (None)
            objects = None
        return objects




