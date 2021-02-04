import seaborn as sns
from Analysis.Constants import *
from pprint import pprint as prettyprint
import Database.Constants as FIELDS
from matplotlib import pyplot as plt
from Analysis.Commons import *
import numpy as np
from wordcloud import WordCloud
from PIL import Image
from scipy.stats import pearsonr
from ImageDetectors.Models.Classification.ColorNaming import name_to_rgb



class Analyzer:
    def __init__(self, database, results_dir = RESULTS_DIR):
        self.db = database
        self.results_dir = results_dir

    # ------------------------------------- ANALYSIS FUNCTIONS --------------------------------------

    def plot_histograms_by_group(self, group_var = FIELDS.TAG, histogram_var = FIELDS.LIKES,
                                 restriction = DEFAULT_RESTRICTION, bins=80, include_fitted_curve=False):
        dir = os.path.join(self.results_dir, HISTOGRAMS_FOLDER_NAME, str(group_var).title(), histogram_var.title())
        if not os.path.isdir(dir):
            os.makedirs(dir)

        restriction = {**restriction, histogram_var: {'$exists': True, '$ne': None}}
        computable_images, total_images = self.count_instances_vs_valid_instances_in_group(group_var=group_var,
                                                                                           restriction=restriction)
        unwind_op = get_unwind_operation(collection=self.db, field=histogram_var, restriction=restriction)
        group_count_op = get_count_operation(collection=self.db, var=histogram_var, restriction=restriction,
                                             numeric_default='$'+histogram_var)
        is_log_scale = field_is_in_log_scale(restriction=restriction, collection=self.db, field=histogram_var)
        if type(bins) is int:
            granularity = 'POWERSOF2' if is_log_scale else 'R10'
            bucket_dict = {'$bucketAuto':{'groupBy': group_count_op,
                                          'buckets': bins,
                                          'granularity' : granularity}}
        elif type(bins) in (list, tuple):
            bucket_dict = {'$bucket': {'groupBy': group_count_op,
                                       'boundaries': bins}}

        groups = (self.db.distinct(group_var, query=restriction) if group_var is not None else [])+[None]
        for group in groups:
            current_restriction = {**restriction, group_var: group} if group is not None else restriction
            query = unwind_op + [{'$match': current_restriction},
                                  bucket_dict]
            histogram = list(self.db.aggregate(query))

            max_bin = bins[bins.index(histogram[-1]['_id'])+1] if type(bins) in (list, tuple) else None
            plot_bucket_histogram(save_dir=dir, histogram=histogram, group=str(group),
                                  x_axis_label=histogram_var.split('.')[-1],
                                  computable_images = computable_images[group], total_images=total_images[group],
                                  include_fitted_curve = include_fitted_curve, log_scale = is_log_scale,
                                  max_bin=max_bin)

    def plot_double_group_histogram(self, group_var_1 = FIELDS.TAG, group_var_2 = FIELDS.COMPLETE_PATH[FIELDS.GENDER],
                                    histogram_var = FIELDS.COMPLETE_PATH[FIELDS.AGE], restriction = DEFAULT_RESTRICTION,
                                    bins=[0,5,10,15,20,25,30,35,40,45,50,80], include_fitted_curve=False, is_log_scale = None):

        dir = os.path.join(self.results_dir, HISTOGRAMS_FOLDER_NAME, group_var_1.title()+' and '+group_var_2.title(),
                           histogram_var.title())
        if not os.path.isdir(dir):
            os.makedirs(dir)

        restriction = {**restriction, histogram_var: {'$exists': True, '$ne': None}}
        computable_images, total_images = self.count_instances_vs_valid_instances_in_group(group_var=group_var_1,
                                                                                           restriction=restriction)
        unwind_op = get_unwind_operation(collection=self.db, field=histogram_var, restriction=restriction)
        group_count_op = get_count_operation(collection=self.db, var=histogram_var, restriction=restriction,
                                             numeric_default='$'+histogram_var)
        if is_log_scale is None:
            is_log_scale = field_is_in_log_scale(restriction=restriction, collection=self.db, field=histogram_var)
        if type(bins) is int:
            granularity = 'POWERSOF2' if is_log_scale else 'R10'
            bucket_dict = {'$bucketAuto':{'groupBy': group_count_op,
                                          'buckets': bins,
                                          'granularity' : granularity}}
        elif type(bins) in (list, tuple):
            bucket_dict = {'$bucket': {'groupBy': group_count_op,
                                       'boundaries': bins,
                                       'default': bins[-1]+1}}
            bins.append(bins[-1]+1)

        groups_1 = (self.db.distinct(group_var_1, query=restriction) if group_var_1 is not None else [])+[None]
        groups_2 = self.db.distinct(group_var_2, query=restriction)
        for group_1 in groups_1:
            current_restriction = {**restriction, group_var_1: group_1} if group_1 is not None else restriction
            histogram, max_bin = {}, -1
            for group_2 in groups_2:
                query = unwind_op + [{'$match': {**current_restriction, group_var_2: group_2}},
                                      bucket_dict]
                histogram[group_2] = list(self.db.aggregate(query))
                try:
                    max_bin = max(max_bin, bins[bins.index(histogram[group_2][-1]['_id'])+1]) if type(bins) in (list, tuple) else None
                except:
                    max_bin = histogram[group_2][-1]['_id']+1
            for normalized in (True, False):
                plot_bucket_double_histogram(save_dir=dir, histograms=histogram, group=str(group_1),
                                              x_axis_label=histogram_var.split('.')[-1],
                                              computable_images = computable_images[group_1],
                                              total_images=total_images[group_1],
                                              include_fitted_curve = include_fitted_curve, log_scale = is_log_scale,
                                              max_bin=max_bin, normalize=normalized)


    def plot_wordcloud_by_tag(self, group_var = FIELDS.TAG, list_of_words_field=FIELDS.ALL_TAGS,
                              restriction = DEFAULT_RESTRICTION, non_relevant_words = NON_RELEVANT_WORDS):
        word_cloud_dir = os.path.join(self.results_dir, WORDCLOUDS_FOLDER_NAME, list_of_words_field.title() + ' - No Pondered')
        histogram_dir = os.path.join(self.results_dir, HISTOGRAMS_FOLDER_NAME, list_of_words_field.title() + ' - No Pondered')
        if not os.path.isdir(word_cloud_dir):
            os.makedirs(word_cloud_dir)
        if not os.path.isdir(histogram_dir):
            os.makedirs(histogram_dir)

        # Adds as restriction that the field should exist
        restriction = {**restriction, list_of_words_field : {'$exists': True},
                       list_of_words_field.split('.')[0] : {'$ne':None}}

        words_concatenation = self.db.aggregate([{'$match': restriction},
                                                 {'$group': {'_id': '$' + group_var,
                                                             'words': {'$push': '$' + list_of_words_field}}},
                                                 {'$project': {group_var : 1,
                                                               'words' : {'$reduce' : {'input' : '$words',
                                                                                       'initialValue' : [],
                                                                                        'in' : {'$concatArrays' : ['$$value', '$$this']}
                                                                                       }}}}])
        _, total_images = self.count_instances_vs_valid_instances_in_group(group_var=group_var,
                                                                                           restriction=restriction)
        words_concatenation = {tag['_id'] : tag['words'] for tag in words_concatenation}
        for group, words in words_concatenation.items():
            words = clean_word_array(array=words, words_to_delete=non_relevant_words | {group})
            fig_name = "Wordcloud of Hashtags related with #{group}".format(group=group.title())
            word_cloud = WordCloud(width=1024, height=1024,background_color=None,
                                   min_font_size=14, max_words=len(words), mode='RGBA',
                                   include_numbers=True, repeat=False,
                                   collocations = False).generate(text=','.join(words))
            Image.fromarray(np.array(word_cloud)).save(os.path.join(word_cloud_dir, fig_name+'.'+PLOTS_FORMAT))
            words, freqs = [], []
            for (word, _), freq, _, _, _ in word_cloud.layout_:
                words.append(word), freqs.append(freq)
            if len(words) <= 20:
                sorted_idx = np.argsort(freqs)[::-1]
                words, freqs = np.array(words)[sorted_idx], np.array(freqs)[sorted_idx]
                colors = get_colors_from_words(words)
                plot_categorical_histogram(save_dir=histogram_dir, height=freqs, group=group,
                                           xlabel=list_of_words_field.split('.')[-1],total_images=total_images[group],
                                           x_ticks=words, colors=colors)


    def plot_point_clouds(self, restriction = DEFAULT_RESTRICTION, group_var = FIELDS.TAG,
                          x_var = FIELDS.LIKES,
                          y_var = FIELDS.COMMENTS, include_fitted_curve = True):
        dir = os.path.join(self.results_dir, POINT_CLOUD_FOLDER_NAME, x_var.title() + ' vs ' + y_var.title())
        if not os.path.isdir(dir):
            os.makedirs(dir)
        restriction = {**restriction, x_var: {'$exists': True, '$ne' : None},
                       y_var: {'$exists': True, '$ne': None}}
        computable_images, total_images = self.count_instances_vs_valid_instances_in_group(group_var=group_var,
                                                                                           restriction=restriction)

        x_count_op = get_count_operation(collection=self.db, var=x_var, restriction=restriction)
        y_count_op = get_count_operation(collection=self.db, var=y_var, restriction=restriction)
        unwind = get_unwind_operation(collection=self.db, field=x_var,restriction=restriction)
        unwind = unwind+get_unwind_operation(collection=self.db, field=x_var,restriction=restriction)
        groups = (self.db.distinct(group_var, query=restriction) if group_var is not None else [])+[None]
        for group in groups:
            current_restriction = {**restriction, group_var:group} if group is not None else restriction
            data = list(self.db.aggregate(unwind+[{'$match' : current_restriction},
                                                  {'$project' : {y_var : y_count_op, x_var: x_count_op}}]))
            x = get_list_for_var_from_mongo_output(mongo_output=data, var=x_var)
            y = get_list_for_var_from_mongo_output(mongo_output=data, var=y_var)
            for remove_outliers in (True, False):
                plot_point_cloud(save_dir=dir, x=x, y=y, group=str(group), x_axis_label=x_var.split('.')[-1],
                                 y_axis_label = y_var.title(), computable_images=computable_images[group],
                                 total_images=total_images[group], include_fitted_rect=include_fitted_curve,
                                 log_scale=False, remove_outliers=remove_outliers)

    def color_histogram(self, restriction=DEFAULT_RESTRICTION, group_var = FIELDS.TAG, color_mode=FIELDS.XKCD_COLORS,
                        object_field = FIELDS.COMPLETE_PATH[FIELDS.FACE_PARTS], part=' skin', bins = 15):

        color_field, part_field = object_field + '.' + color_mode, object_field + '.' + FIELDS.PARTS

        dir = os.path.join(self.results_dir, COLOR_PLOT_FOLDER_NAME,
                           '{obj}_{part}-{color}'.format(obj=object_field, part=part, color=color_mode),
                           '{bins}Bins'.format(bins=bins))

        if not os.path.isdir(dir):
            os.makedirs(dir)
        restriction = {**restriction, object_field : {'$exists': True, '$ne': None},
                       color_field+'.0' : {'$exists': True}, part_field : part}
        unwind_op = get_unwind_operation(collection=self.db, field=object_field + '.' + FIELDS.PARTS,
                                     restriction=restriction)

        _, total_images = self.count_instances_vs_valid_instances_in_group(group_var=group_var,
                                                                           restriction=restriction)

        groups = (self.db.distinct(group_var, query=restriction) if group_var is not None else [])+[None]
        for group in groups:
            current_restriction = {**restriction, group_var: group} if group is not None else restriction

            query = unwind_op + [{'$match' : current_restriction},
                                 {'$project' : {'color' : {'$arrayElemAt' : ['$'+color_field,
                                                                            {'$indexOfArray' : ['$'+part_field, part]}]
                                                          }}}]

            colors = self.db.aggregate(query)
            color_names = get_list_for_var_from_mongo_output(mongo_output=colors, var='color')
            color_clusters, frequency = get_colors_histogram(color_names_list=color_names,bins=bins)
            plot_categorical_histogram(save_dir=dir, colors=color_clusters, height=frequency, group=group,
                                       xlabel= part + ' Color', total_images=total_images[group])



    # ---------------------------------------------- AUXILIARS ------------------------------------------

    def count_instances_vs_valid_instances_in_group(self, group_var, restriction):
        computable_images, total_images = {}, {}
        if group_var is not None:
            unwind = get_unwind_operation(collection=self.db,field=group_var, restriction=restriction)
            computable_images = self.db.aggregate(unwind + [{'$match': restriction},
                                                            {'$group': {'_id': '$' + group_var,
                                                                        'count': {'$sum': 1}
                                                                       }}])
            computable_images = {group['_id']: group['count'] for group in computable_images}
            total_images = self.db.aggregate(unwind + [{'$group': {'_id': '$' + group_var,
                                                                   'count': {'$sum': 1}
                                                                  }}])
            total_images = {group['_id']: group['count'] for group in total_images}

        computable_images[None] = self.db.count(restriction)
        total_images[None] = self.db.count()
        return computable_images, total_images

# ------------------------------------- PLOTTING FUNCTIONS -----------------------------------------

def plot_point_cloud(save_dir, x, y, group, x_axis_label, y_axis_label,
                     computable_images, total_images, include_fitted_rect = False,
                     log_scale = False, remove_outliers = True, anomaly_quantile = 0.02, kde_bw=2.5):
    fig_name = "{x} vs {y} for {group}".format(x=x_axis_label.title(),
                                                            y = y_axis_label.title(),
                                                            group=group_clean_str(group))
    if not remove_outliers:
        anomal_x, anomal_y = detect_anomalies(x, y, quantile=anomaly_quantile, remove_anomalies=False, bw=kde_bw)
        try:
            sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_method='silverman', bw_adjust=kde_bw, levels=20,
                        thresh=anomaly_quantile)
        except:
            sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_method='silverman', bw_adjust=kde_bw,
                        thresh=anomaly_quantile)
        pearson = round(pearsonr(x, y)[0], ndigits=3)
        plt.plot(x, y, 'bo', alpha=0.2, label='Data (pearson corr: {pearson})'.format(pearson=pearson))
        plt.plot(anomal_x, anomal_y, 'ro', alpha = 0.4, label='Outliers')
    else:
        x, y = detect_anomalies(x, y, quantile=anomaly_quantile, remove_anomalies=True, bw=kde_bw)
        try:
            sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_method='silverman', bw_adjust=kde_bw, levels=20,
                        thresh=anomaly_quantile)
        except:
            sns.kdeplot(x=x, y=y, cmap="Reds", shade=True, bw_method='silverman', bw_adjust=kde_bw,
                        thresh=anomaly_quantile)
        pearson = round(pearsonr(x, y)[0], ndigits=3)
        plt.plot(x, y, 'bo', alpha=0.2, label='Data (pearson corr: {pearson})'.format(pearson=pearson))

    plt.xlim(0, max(x))
    plt.ylim(0, max(y))

    if include_fitted_rect:
        full_x_range = np.arange(min(x), max(x), 0.2)
        spline, slope = fitted_spline(x=x, y=y, objective_x_range=full_x_range,model=LINEAR, return_parameter='slope')
        plt.plot(full_x_range, spline, 'r--', label = 'Linear Model. Slope: '+str(round(slope, ndigits=3)))
    if log_scale:
        plt.xscale('log', basex=2)
    plt.xlabel(x_axis_label.title())
    plt.ylabel(y_axis_label.title())
    plt.legend()
    plt.title(fig_name + ' (Valid Posts: {comp}/{total} ({perc} %))'.
              format(comp=computable_images, total=total_images,
                     perc=round((computable_images / total_images) * 100, ndigits=2)))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name.replace('#', '') + (' - Without Outliers' if remove_outliers else '') + '.' + PLOTS_FORMAT))
    plt.close()

def plot_bucket_histogram(save_dir, histogram, group, x_axis_label,
                          computable_images, total_images, include_fitted_curve = False,
                          log_scale = False, max_bin=80):
    fig_name = "Hist of {hist} for {group}".format(hist=x_axis_label.title(),
                                                 group=group_clean_str(group))
    if 'min' in histogram[0]:
        x_mins = [bar['_id']['min'] for bar in histogram]
        x_max = [bar['_id']['max'] for bar in histogram]
        x_width = [(max - min) * 0.95 for min, max in zip(x_mins, x_max)]
    else:
        x_mins = [bar['_id'] for bar in histogram]
        x_max = [bar['_id'] for bar in histogram[1:]]+[max_bin]
        separation = min([(max - min) - 0.5 for min, max in zip(x_mins, x_max)]) * 0.05
        x_width = [(max - min) - separation for min, max in zip(x_mins, x_max)]

    x = [(max + min) / 2 for min, max in zip(x_mins, x_max)]
    y = [bar['count'] for bar in histogram]
    plt.bar(x=x, height=y, width=x_width)
    if include_fitted_curve:
        full_x_range = np.arange(0, x[-1])
        spline = fitted_spline(x=x, y=y, objective_x_range=full_x_range, model=POLYNOMIAL)
        plt.plot(full_x_range, spline, 'r--')
    if log_scale:
        plt.xscale('log', basex=2)
    plt.xlabel(x_axis_label.title())
    plt.ylabel('Amount')
    plt.xticks(x_mins + [x_max[-1]], labels=x_mins + [x_max[-1]], rotation=45)
    plt.title(fig_name + ' (Posts with {label}: {comp}/{total} ({perc} %))'.
              format(label=x_axis_label.title(),
                     comp=computable_images, total=total_images,
                     perc=round((computable_images / total_images) * 100, ndigits=2)))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name.replace('#', '') + '.' + PLOTS_FORMAT))
    plt.close()

def plot_bucket_double_histogram(save_dir, histograms, group, x_axis_label,
                          computable_images, total_images, include_fitted_curve = False,
                          log_scale = False, max_bin=80, normalize = True):
    fig_name = "{norm}Hist of {hist} for {group}".format(norm=('Norm ' if normalize else ''),
                                                         hist=x_axis_label.title(), group=group_clean_str(group))
    h_len = len(histograms)
    if h_len == 0: return
    for i, (label, histogram) in enumerate(histograms.items()):
        if type(histogram[0]['_id']) is dict and 'min' in histogram[0]['_id']:
            max_bin = histogram[-1]['_id']['max']
            histogram = [{'_id' : bar['_id']['min'],  'count' : bar['count']} for bar in histogram]


        x_mins = [bar['_id'] for bar in histogram]
        x_max = [bar['_id'] for bar in histogram[1:]]+[max_bin]
        separation = min([(max - min) - 0.5 for min, max in zip(x_mins, x_max)]) * 0.05
        if log_scale:
            x_width = [((max - min)/h_len) - separation for min, max in zip(x_mins, x_max)]
        else:
            separation = 0
            x_width = [((max - min) / h_len) for min, max in zip(x_mins, x_max)]
        if i == h_len-1: pad = -separation
        elif i != 0: pad = 0
        else: pad = separation
        #  intepolation
        x = [((min*(h_len-i)+(max*i))/h_len)+width/2+pad/2+separation for min, max, width in zip(x_mins, x_max, x_width)]
        y = [bar['count'] for bar in histogram]
        y_to_plot = y/np.sum(y) if normalize else y
        plt.bar(x=x, height=y_to_plot, width=x_width, label=label, color=get_color_for_word(word=label))
        if include_fitted_curve:
            full_x_range = np.arange(x_mins[0], x[-1])
            spline = fitted_spline(x=x, y=y, objective_x_range=full_x_range, model=POLYNOMIAL)
            plt.plot(full_x_range, spline, '--', label=label+' Polynomial Approx')

    plt.legend()

    if log_scale:
        plt.xscale('log', basex=2)
    plt.xlabel(x_axis_label.title())
    plt.ylabel('Amount' if not normalize else 'Percentage')
    plt.ylim(0)
    plt.xticks(x_mins + [x_max[-1]], labels=x_mins + [x_max[-1]], rotation=45)
    plt.title(fig_name + ' (Posts with {label}: {comp}/{total} ({perc} %))'.
              format(label=x_axis_label.title(),
                     comp=computable_images, total=total_images,
                     perc=round((computable_images / total_images) * 100, ndigits=2)))
    plt.tight_layout()
    if normalize:
        save_dir = os.path.join(save_dir, 'Normalized')
    else:
        save_dir = os.path.join(save_dir, 'Not Normalized')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, fig_name.replace('#', '') + '.' + PLOTS_FORMAT))
    plt.close()

def plot_categorical_histogram(save_dir, height, group, xlabel, total_images, colors=DEFAULT_COLOR, x_ticks = [],
                               include_fitted_curve = False):

    fig_name = "Hist for {xlabel} in {group}".format(xlabel=xlabel.title().strip(), group=group_clean_str(group))

    plt.bar(x=range(len(height)), height=height, color=colors)
    if include_fitted_curve:
        full_x_range = np.arange(0, len(height),0.2)
        spline = fitted_spline(x=np.arange(0, len(height)), y=colors, objective_x_range=full_x_range)
        plt.plot(full_x_range, spline, 'r--')
    plt.xlabel(xlabel.title().strip())
    plt.ylabel('Frequency')

    plt.xticks(ticks=range(len(height)), labels=x_ticks, rotation=45)
    plt.title(fig_name + ' ({comp} {type} from {total} Images)'.
              format(comp=np.sum(height), type=xlabel.title().strip()+'s', total=total_images))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fig_name.replace('#', '') + '.' + PLOTS_FORMAT))
    plt.close()

def pprint(result):
    for sample in result:
        prettyprint(sample)

