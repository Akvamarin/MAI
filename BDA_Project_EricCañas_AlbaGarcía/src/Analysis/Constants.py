import os
from Database import Constants as FIELDS

RESULTS_DIR = os.path.join('Analysis', 'Results')
HISTOGRAMS_FOLDER_NAME = 'Histograms'
WORDCLOUDS_FOLDER_NAME = 'Wordclouds'
POINT_CLOUD_FOLDER_NAME = 'PointClouds'
COLOR_PLOT_FOLDER_NAME = 'ColorHistograms'
GRAPHS_FOLDER_NAME = 'Graphs'

MIN_POS, MAX_POS = 0, 1
AVG_LIKES_RANGE = (0, 60)
SEMI_POPULAR_LIKES_RANGE = (60, 150)
POPULAR_LIKES_RANGE = (150, 500)
SEMI_INFLUENCER_LIKES_RANGE = (500, 2000)
INFLUENCER_LIKES_RANGE = (2000, 10000)
SUPER_INFLUENCER_LIKES_RANGE = (10000, 1000000000)


DEFAULT_RESTRICTION = {FIELDS.TOP_POST : False, FIELDS.LIKES : {'$ne' : None}, FIELDS.IS_VIDEO : False}
NON_RELEVANT_WORDS = {'floor', 'wall', 'ceiling', ' background', None}

WORD_ARRAY_FIELDS = (FIELDS.ALL_TAGS, FIELDS.COMPLETE_PATH[FIELDS.GENERAL_PARTS],
                     FIELDS.COMPLETE_PATH[FIELDS.OBJECT], FIELDS.COMPLETE_PATH[FIELDS.INSTAGRAM_DESCRIPTORS])

PLOTS_FORMAT = 'png'


DEFAULT_COLOR = 'xkcd:bright blue'
ASSOCIATED_COLORS = {'sad' : 'xkcd:deep blue', 'happy' : 'xkcd:yellowish', 'neutral' : 'xkcd:grey/blue',
                     'angry' : 'xkcd:carmine', 'fear' : 'xkcd:lawn green', 'surprise' : 'xkcd:topaz',
                     'female' : 'xkcd:pale purple', 'male' : 'xkcd:baby blue'}