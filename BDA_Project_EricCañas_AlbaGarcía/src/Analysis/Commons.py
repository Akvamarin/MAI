# -*- coding: utf-8 -*-
from Database import Constants as FIELDS
from Analysis.Constants import *
from scipy.optimize import curve_fit
import numpy as np
from astropy.modeling import models, fitting
from sklearn.neighbors import KernelDensity
from nltk.cluster.kmeans import KMeansClusterer #NLTK instead sklearn because it allows custom distances (for eCIE2000)
from nltk.cluster.util import euclidean_distance
from ImageDetectors.Models.Classification.ColorNaming import *
from sklearn.manifold import TSNE
from umap import UMAP
from warnings import warn


GAUSSIAN, LINEAR, POLYNOMIAL, VOIGT = 'gaussian', 'linear', 'polinomial', 'voigt'
MODELS = {GAUSSIAN : models.Gaussian1D, LINEAR : models.Linear1D,
          POLYNOMIAL : lambda : models.Polynomial1D(degree=8, domain=(0,1), window=(0,1)),
          VOIGT : models.Voigt1D}
FITTING_FUNCTIONS = {GAUSSIAN : fitting.LevMarLSQFitter, LINEAR : fitting.LinearLSQFitter,
                     POLYNOMIAL : fitting.LevMarLSQFitter, VOIGT : fitting.LevMarLSQFitter}

def get_restriction(likes_range = None, allow_top_post = True, only_posts_with_one_face = True, restriction = {}):
    if likes_range is not None:
        restriction[FIELDS.LIKES] = {'$gte' : likes_range[MIN_POS], '$lt' : likes_range[MAX_POS]}
    if not allow_top_post:
        restriction[FIELDS.TOP_POST] = False
    if only_posts_with_one_face:
        restriction[FIELDS.COMPLETE_PATH[FIELDS.FACES]] = {'$size' : 1}
    return restriction


def fitted_spline(x, y, objective_x_range, model = GAUSSIAN, add_0_at_queues = False, return_parameter = None):
    assert return_parameter in (None, False, 'slope', 'Slope')
    model_name = model.lower()
    if model_name in MODELS.keys():
        model, fitting_function = MODELS[model_name](), FITTING_FUNCTIONS[model_name]()
    else:
        raise NotImplementedError("Model {m} doesn't exist. Available models are: {ms}".format(m=model,
                                                                                               ms=list(MODELS.keys())))
    if add_0_at_queues:
        first_queue = np.arange(0, int(x[0]))
        second_queue = np.arange(int(x[-1]+(x[-1]-x[-2]+x[0])), int(x[-1]*2))
        x = list(first_queue)+list(x)+list(second_queue)
        y = [y[0] for _ in first_queue]+list(y)+[y[-1] for _ in second_queue]
    max_value = max(max(y), max(x))
    t = fitting_function(model, np.array(x)/max_value, np.array(y)/max_value)
    y_line = t(np.array(objective_x_range)/max_value)*max_value
    if not return_parameter:
        return y_line
    else:
        if hasattr(t, return_parameter.lower()):
            return y_line, getattr(t, return_parameter.lower()).value
        else:
            raise NotImplementedError('{param} is not a valid parameter for {model} model'.format(param=return_parameter,
                                                                                                  model=model_name))

def clean_word_array(array, words_to_delete = []):
    """
    Clean a words array. It is, putting all words in lowercase, substituting spaces by underscores and deleting
    all words in the list of words_to_delete.
    :param array: List of str. List of words
    :param words_to_delete: List of str. List of words to delete.
    :return: List of str. array but with all words in a more correct format.
    """
    if type(words_to_delete) not in (list, tuple, set):
        words_to_delete = [words_to_delete]
    return [word.lower().replace(' ', '_') for word in array if word not in words_to_delete]

def group_clean_str(group):
    if group in os.listdir(FIELDS.IMAGE_DATASET_PATH):
        return '#'+group.title()
    elif group == 'None':
        return 'Dataset'

def get_count_operation(collection, var, restriction, numeric_default = True):
    """
    Return the operation that should be used for counting an element in a query. Usually True if it is directly an int
    or {$size : var} if it is an array element.
    :param collection: Pymongo collection. Collection where to check
    :param var: String. Variable to check
    :param restriction: Dict. Restrictions to apply for finding a correct example of the variable
    :param numeric_default: Dict or bool. Value to return if the value is already a number. Default: True
    :return: Dict or bool. Operation that should be used for counting the content of variable var.
    """
    var_example = collection.find_one(restriction, {var : True})
    for key in var.split('.'):
        if type(var_example) in (list, tuple, set):
            var_example = var_example[0]
        var_example = var_example[key]
    if type(var_example) in (int, float):
        return numeric_default
    elif type(var_example) in (list, tuple):
        return {'$size':'$' + var}
    else:
        raise NotImplementedError('{var} is of type {type} and can not be counted'.format(var=var,
                                                                                          type=type(var_example)))

def get_list_for_var_from_mongo_output(mongo_output, var):
    """
    Given a mongodb result (iterable) generates a list with all the values of the field var.
    :param mongo_output: Iterable of dictionaries. MongoDB result or list (it should be reused later) containing the
                         results of a mongo query where var was used.
    :param var: string. Field contained in the query (can be in dot notation)
    :return: List. List with all the values of the field var.
    """
    out = []
    for sample in mongo_output:
        for key in var.split('.'):
            if type(sample) in (list, tuple):
                if len(sample) > 1:
                    warn("List field with more than one entry, but only first computed. You should use unwind.")
                sample = sample[0]
            else:
                pass
                #raise ValueError("List with multiple entries")
            sample = sample[key]
        out.append(sample)
    return out

def detect_anomalies(x, y, quantile=0.025, remove_anomalies = True, bw=2.5):
    """
    Detect the anomalies within X, Y. If remove_anomalies is True, return the X, Y data cleaned if False,
    return the anomaly data points.
    :param x: 1-d numeric array. X data
    :param y: 1-d numeric array. Y data
    :param quantile: float. Inferior quantile for considering a point an anomaly (in function of its kernel density)
    :param return_anomalies: boolean. If True, return the X, Y data cleaned else  return the anomaly data points.
                            default. True
    :return: If remove_anomalies is True, return the X, Y data cleaned otherwise, return the anomaly data points.
    """
    assert len(x) == len(y)
    data = np.concatenate((x,y)).reshape(2, len(x)).T
    kernel_density = KernelDensity(kernel=GAUSSIAN, bandwidth=bw).fit(data)
    scores =  kernel_density.score_samples(data)
    if remove_anomalies:
        out = data[scores >= np.quantile(scores, quantile)]
    else:
        out = data[scores < np.quantile(scores, quantile)]
    return out[:, 0], out[:, 1]

def get_unwind_operation(collection, field, restriction):
    """
    Returns the unwind operations needed for de-nestcollection.find_one(restriction)ing the content of a given field,
    depending on the array fields which it contains.
    :param db: Pymongo collection. Collection where to check.
    :param field: String. Field to unwind in dot notation
    :return: List of dict. List of operations needed for de-nesting the content of the given field.
    """
    unwind_ops = []
    valid_sample = collection.find_one(restriction)
    for i, sub_field in enumerate(field.split('.')[:-1]):
        valid_sample = valid_sample[sub_field]
        if type(valid_sample) in [list, tuple, set]:
            unwind_ops.append({'$unwind' : '$'+'.'.join(field.split('.')[:i+1])})
            valid_sample = valid_sample[0]
    return unwind_ops

def get_colors_histogram(color_names_list, bins=10, color_mode ='lab', reordering='umap', kmeans_repeats=2):
    """
    :param lab_colors_list:
    :return:
    """
    assert color_mode.lower() in ('lab', 'rgb', 'hsv')
    assert reordering.lower() in ('tsne', 'umap', 'hsv_h', 'hsv_s', 'hsv_v')
    if color_mode.lower() == 'rgb':
        color_array = np.array([name_to_rgb(name_color=color) for color in color_names_list], dtype=np.float32)
        distance = euclidean_distance
    elif color_mode.lower() == 'lab':
        color_array = np.array([name_to_lab(name_color=color) for color in color_names_list], dtype=np.float32)
        distance = deltaECIE2000
    elif color_mode.lower() == 'hsv':
        color_array = np.array([name_to_hsv(name_color=color) for color in color_names_list], dtype=np.float32)
        distance = hsv_distance
    kmeans = KMeansClusterer(num_means=bins, distance=distance, repeats=kmeans_repeats, avoid_empty_clusters=True)
    clusters = kmeans.cluster(vectors=color_array, assign_clusters=True)
    clusters, histogram = np.unique(clusters, return_counts=True)
    colors = np.array(kmeans.means(), dtype=np.float32)[clusters]
    if reordering.lower() == 'tsne':
        tsne = TSNE(n_components=1, metric=distance)
        transformed_location = tsne.fit_transform(X=colors).squeeze()
    elif reordering.lower() == 'umap':
        umap = UMAP(n_neighbors=max(len(clusters)//4,2), n_components=1, metric=distance)
        transformed_location = umap.fit_transform(X=colors).squeeze()
    else:
        index = 'hsv'.index(reordering.lower()[-1])
        if color_mode.lower() == 'lab':
            transformed_location = [lab_to_hsv(lab_color=color)[index] for color in colors]
        elif color_mode.lower() == 'rgb':
            transformed_location = [rgb_to_hsv(rgb_color=color)[index] for color in colors]
        elif color_mode.lower() == 'hsv':
            transformed_location = [color[index] for color in colors]
    sorted_idx = np.argsort(transformed_location)
    colors, histogram = list(colors[sorted_idx]), histogram[sorted_idx]
    if color_mode.lower() == 'lab':
        colors = [lab_to_rgb(lab_color=color) for color in colors]
    elif color_mode.lower() == 'hsv':
        colors = [hsv_to_rgb(hsv_color=color) for color in colors]
    return colors, histogram

def field_is_in_log_scale(restriction, collection, field, log_threshold = 256):
    return collection.find_one({**restriction, field : {'$gte': log_threshold}}) is not None

def get_colors_from_words(words, default_color = None):
    colors = []
    for word in words:
        word = word.lower()
        if word not in ASSOCIATED_COLORS:
            return default_color
        else:
            colors.append(ASSOCIATED_COLORS[word])
    return colors

def get_color_for_word(word, default_color = None):
    return ASSOCIATED_COLORS.get(word.lower(), default_color)




