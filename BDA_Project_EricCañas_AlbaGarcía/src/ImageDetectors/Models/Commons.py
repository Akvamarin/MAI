from ImageDetectors.Models.Constants import *
import numpy as np
from skimage.transform import resize



RANDOM_SEED = 0

def get_class_names(dataset = ADE20K, csv_path = BASE_DATASET_DESCRIPTION_PATH):
    """
    Get the class names of a given dataset, reading its csv
    :param dataset: String. Name of the dataset. Its CSV shoulf be located at the given csv_path.
    :param csv_path: String path. Path where all the csv files containing the dataset class names are located.
    :return: Numpy Array of type '<U128'. Array with the class names of the given dataset
    """
    # Check that the information of this dataset is available
    if not os.path.isfile(os.path.join(csv_path, dataset+'.csv')):
        raise RuntimeError("Class names of {df} are not available.\n Unique available datasets are: {aval}"
                           .format(df=dataset, aval=os.listdir(csv_path)))
    # Process the csv
    info_path = os.path.join(csv_path, dataset + '.csv')
    info = np.genfromtxt(fname=info_path, dtype='<U128',skip_header=True,delimiter=',')
    info = info[..., -1]
    classes = [classes.split(MULTIPLE_CLASSES_SEPARATOR)[0] for classes in info]
    # Return the class names formatted as numpy for faster association.
    return np.array(classes, dtype='<U128')

def get_color_codes(num_classes):
    """
    Return a list of num_classes rgb colors that are deterministic.
    :param num_classes: Int. Amount of rgb codes to retrieve
    :return: Array of RGB uint8 color codes
    """
    np.random.seed(RANDOM_SEED)
    return np.random.randint(0,np.iinfo(np.uint8).max, size=(num_classes, len('RGB'))).astype(np.uint8)

def fuse_images_without_background(img, img_to_insert, img_to_insert_segmentation_mask, box):
    """
    Fuse two images, without pasting the background of img_to_insert into img.
    Background id must be 0 in the img_to_insert_segmentation_mask, in order to erase it.
    :param img: Numpy array image. Image where to paste img_to_insert without the background
    :param img_to_insert: Numpy array image. Image to paste in img but avoiding the background
    :param img_to_insert_segmentation_mask: Numpy array of same size than img_to_insert. Segmentation mask of
                                            img_to_insert, where the background is labeled as 0.
    :param box: Tuple of int in the form x1, y1, x2, y2. Location on the img space where to paste img_to_insert
    :return: Numpy array image with the same format as img. It contains the non background part of img_to_insert pasted.
    """
    x1, y1, x2, y2 = box
    # Take the patch of the original image that will suffer modification
    original_img_patch = img[y1:y2, x1:x2]
    # Extract a boolean mask containing the background
    background_mask = resize(image=img_to_insert_segmentation_mask, output_shape=original_img_patch.shape[:2],preserve_range=True,
           anti_aliasing=False).astype(np.bool)
    # Paste the non background part of img_to_insert in this patch
    original_img_patch[background_mask] = img_to_insert[background_mask]
    # Put again the modified patch into img
    img[y1:y2, x1:x2] = original_img_patch
    return img
