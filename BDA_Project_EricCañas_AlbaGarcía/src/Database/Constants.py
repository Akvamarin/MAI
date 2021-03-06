import os

# DATABASE FIELDS
ID, TAG, DATE, LIKES, COMMENTS, TOP_POST, MENTIONED_USERS, ALL_TAGS, IS_CAROUSEL, IS_VIDEO, INSTAGRAM_DESCRIPTORS, IMAGE_FEATURES = \
    '_id', 'tag', 'date', 'likes', 'comments', 'is_a_top_post', 'mentioned_users', 'all_tags',  'is_carousel', 'is_video',\
    'instagram_descriptors', 'image_features'

FACES, FACE = 'faces', 'face'
SEGMENTATION = 'segmentation'
GENERAL_PARTS = 'general_parts'
OBJECT_COUNTING = 'object_counting'
OBJECT, COUNT = 'objects', 'count'
GENDER, AGE, EMOTION = 'gender', 'age', 'emotion'
FACE_PARTS = 'face_parts'
PARTS, AREAS, BASIC_COLORS, XKCD_COLORS = 'parts', 'areas', 'basic_colors', 'xkcd_colors'
FACE_BASIC_COLORS, FACE_XKCD_COLORS = BASIC_COLORS, XKCD_COLORS
PARTS_BASIC_COLORS, PARTS_XKCD_COLORS = BASIC_COLORS, XKCD_COLORS

TOTAL_AREA = 'total_area'
COMPLETE_PATH = {FACES : IMAGE_FEATURES+'.'+FACES,
                 GENERAL_PARTS : IMAGE_FEATURES+'.'+GENERAL_PARTS+'.'+PARTS,
                 PARTS_BASIC_COLORS : IMAGE_FEATURES+'.'+GENERAL_PARTS+'.'+BASIC_COLORS,
                 PARTS_XKCD_COLORS : IMAGE_FEATURES+'.'+GENERAL_PARTS+'.'+XKCD_COLORS,
                 OBJECT : IMAGE_FEATURES + '.' + OBJECT_COUNTING + '.' + OBJECT,
                 INSTAGRAM_DESCRIPTORS : IMAGE_FEATURES + '.' + INSTAGRAM_DESCRIPTORS,
                 FACE_PARTS : IMAGE_FEATURES+'.'+FACES+'.'+FACE_PARTS,
                 FACE_BASIC_COLORS : IMAGE_FEATURES+'.'+FACES+'.'+FACE_PARTS+'.'+FACE_BASIC_COLORS,
                 GENDER : IMAGE_FEATURES+'.'+FACES+'.'+GENDER,
                 AGE : IMAGE_FEATURES+'.'+FACES+'.'+AGE,
                 EMOTION : IMAGE_FEATURES+'.'+FACES+'.'+EMOTION}

NUMERIC_FIELDS = [COMPLETE_PATH[FACES], COMPLETE_PATH[AGE], LIKES, COMMENTS]
CATECORICAL_FIELDS = [COMPLETE_PATH[GENDER], COMPLETE_PATH[EMOTION], TAG]
DESCRIPTOR_FIEDS = [COMPLETE_PATH[INSTAGRAM_DESCRIPTORS], ALL_TAGS, COMPLETE_PATH[GENERAL_PARTS],
                    COMPLETE_PATH[OBJECT], COMPLETE_PATH[FACE_PARTS]]
COLOR_PARTS = [' skin', ' hair']


# COLLECTION META INFORMATION
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 27017
DEFAULT_DB = 'InstagramCrawler'
IMAGE_DATASET_PATH = os.path.join('Database', 'Dataset')
IMAGES_SAVE_FORMAT = 'png'
