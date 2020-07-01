from django.apps import AppConfig
from django.conf import settings
import os
import pickle
class PredictorConfig(AppConfig):
    # create path to models
    path = os.path.join(settings.MODELS, 'knnmodel.p')
 
    # load models into separate variables
    # these will be accessible via this class
    with open(path, 'rb') as pickled:
       data = pickle.load(pickled)
    
    sc_x= data['standard_scalar']
    knn = data['knn']

class CnnConfig(AppConfig):
	path = os.path.join(settings.MODELS,'model_best3.pt')