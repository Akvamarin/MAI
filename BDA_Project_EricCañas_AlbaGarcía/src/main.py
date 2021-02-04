from Crawler.HashtagsCrawler.HashtagCrawler import HashtagCrawler
from Database.HashtagDB.HashtagDB import HashtagDB
from ImageDetectors.VisualAnalysisPipeline import Pipeline
from Analysis.Analyzer import Analyzer
from Database import Constants as FIELD
from Analysis.GraphAnalysis.InsTagNetAnalyzer import InsTagNetAnalyzer
from itertools import combinations

MODES = ['DATA GATHERING', 'DATA ANALYSIS', 'TAGNET']
MODE = 'DATA GATHERING'

if __name__ == '__main__':
    with HashtagDB() as db:
        if MODE.upper() == 'DATA GATHERING':
            crawler = HashtagCrawler(database=db, feature_extraction_pipeline=Pipeline())
            crawler.collect()
            crawler.close()
        elif MODE.upper() == 'DATA ANALYSIS':
            analyzer = Analyzer(database=db)
            # Word Clouds generation
            #analyzer.plot_double_group_histogram(histogram_var=FIELD.COMMENTS, bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], is_log_scale=False)
            analyzer.plot_point_clouds(x_var=FIELD.COMPLETE_PATH[FIELD.AGE], y_var=FIELD.LIKES)
            analyzer.plot_point_clouds(x_var=FIELD.COMPLETE_PATH[FIELD.AGE], y_var=FIELD.COMMENTS)
            """
            for field in FIELD.DESCRIPTOR_FIEDS:
                try: tagnet = InsTagNetAnalyzer(collection=db, base_field=field)
                except: continue
                for min_rel in (0.1,0.2,0.3,0.4,0.5):
                    try: tagnet.plot_graph(minimum_ploteable_weight_percentage=min_rel)
                    except: pass
            """
            """
            for bins in (15, 20, 25):
                analyzer.color_histogram(group_var=None, part=' skin', bins=bins)
                analyzer.color_histogram(group_var=FIELD.COMPLETE_PATH[FIELD.GENDER], part=' skin', bins=bins)
                try: analyzer.color_histogram(group_var=FIELD.COMPLETE_PATH[FIELD.GENDER], part=' hair', bins=bins)
                except: pass
            """


        elif MODE.upper() == 'TAGNET':
            tagnet = InsTagNetAnalyzer(collection=db)
            tagnet.plot_graph()

        else:
            raise NotImplementedError("Available modes are only: {modes}".format(modes=MODES))