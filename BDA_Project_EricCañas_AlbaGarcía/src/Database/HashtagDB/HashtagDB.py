from Database.MongoDB import MongoDB
from Database.Constants import *
from pprint import pprint
from PIL import Image
import numpy as np

class HashtagDB(MongoDB):
    """
    Child class of MongoDB, which implements the collection and offer all the functions that are specifically useful
    for an Instagram Hashtag crawler.
    """
    def __init__(self, collection_name=DEFAULT_DB, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 image_dataset_path = IMAGE_DATASET_PATH):
        """
        Concrete implementation of the Hashtag MongoDB Database. It also checks that the information within the db
        and the unidentifiable dataset of segmentations maps is consistent. If it is not, delete the inconsistencies.
        :param collection_name: String. Name of the collection where to save
        :param host: String. Ip of the host where the db is located
        :param port: Int. Port from where the db service is offered
        :param image_dataset_path: String. Path where to the simplified unidentifiable images are saved for references.
        """
        super().__init__(collection_name=collection_name, host=host, port=port)
        self.collection = self.db.hashtag_collection
        self.image_dataset_path = image_dataset_path
        # Find if there is any inconsistency between the db and the dataset and solve them
        self.clean_db()

        # Override the pymongo functions for making the collection management transparent
        self.find = self.collection.find
        self.find_one = self.collection.find_one
        self.insert = self.collection.insert
        self.aggregate = self.collection.aggregate
        self.distinct = self.collection.distinct
        self.count = self.collection.count



    def find_images(self, sentence_dict, labels_to_return=None, format='numpy'):
        """
        Execute a find operation in the db, but returning the images which match the sentence instead of the db fields.
        If labels to return is not None, also return those fields
        :param sentence_dict: MongoDB find sentence.
        :param labels_to_return: Iterable with the labels to return. If None, do not return this second argument. Default None
        :param format: str. 'numpy', 'np' or 'PIL'. Format in which return the images.
        :return: Iterable with each resulting image. If labels to return is not None --> tuple (Image, dictionary),
                where the dictionary contains the value of the fields in labels to return
        """
        if format.lower() not in ['numpy', 'np', 'pil']:
            raise NotImplementedError("Unique valid formats are 'numpy' and 'PIL'")

        labels_to_search = set() if labels_to_return is None else set(labels_to_return)
        # Ensure to return the ID and the TAG during the search, for linking to the image in the dataset.
        labels_to_search.update({ID, TAG})

        for document in self.collection.find(sentence_dict, labels_to_search):
            # Open the image and convert it to RGB
            image = Image.open(os.path.join(self.image_dataset_path, document[TAG], document[ID]+'.'+IMAGES_SAVE_FORMAT))
            image = image.convert('RGB')
            if format.lower() in ['numpy', 'np']:
                image = np.array(image)
            # Yield the image
            if labels_to_return is None:
                yield image
            # Yield the image and its requested labels
            else:
                yield image, {label : document[label] for label in labels_to_return}


    def clean_db(self):
        """
        Erase from the db every instance whose image is lost
        :param dataset_path: str path. Path of the image dataset which should match with the information within
         the collection
        """

        saved_images = get_ids_of_saved_images(dataset_dir=self.image_dataset_path)
        if len(saved_images) > 0:
            images_lost = [doc[ID] for doc in self.collection.find({}, {ID, TAG})
                           if doc[TAG] not in saved_images or not doc[ID] in saved_images[doc[TAG]]]
            if len(images_lost) > 0:
                if input("{n} instances of the dataset have no correspondent image at {dpath}. Delete them? [Yes|No]"
                                 .format(n=len(images_lost), dpath=self.image_dataset_path)).lower() in ['y', 'yes']:
                    deleted_documents = self.collection.delete_many({ID: {'$in': images_lost}}).deleted_count
                    print("{n} documents deleted".format(n=deleted_documents))
                else:
                    print("They were no deleted")
        else:
            if input("Dataset at {dpath} in empty. Delete the associated collection? [Yes|No]"
                             .format(dpath=self.image_dataset_path)).lower() in ['y', 'yes']:
                self.collection.drop()
            else:
                print("Database was not deleted")

def get_ids_of_saved_images(dataset_dir=IMAGE_DATASET_PATH):
    """
    Get a dictionary with the id of the images that are inside the dataset for each tag. Useful for inducing by hashing
    if an image is repeated or not
    :param dataset_dir: str. Directory where of the associated image dataset
    :return: Dictionary of dictionaries. Key1 TAG, key2 ID.
    """
    out = {}
    if os.path.isdir(dataset_dir):
        for element in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, element)
            if os.path.isdir(path):
                # Extract recursively the content of the directory
                out[element] = get_ids_of_saved_images(dataset_dir=path)
            elif os.path.isfile(path):
                # Value does not matter since it only wants to make the key hashable in O(1) (save without the extension)
                out[os.path.splitext(element)[0]] = True
    return out