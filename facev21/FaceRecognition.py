import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from facev21.model import VggFace
from facev21.face_detection import Mediapipe, FaceDetector, RetinaFace, OpenCv
from facev21.extension import distance, functions

def build_model(model_name):
    
    model_obj = {} # singleton design pattern
    
    models = {
        'VGG19': VggFace.load_model
    }
    
    if not 'model_obj' in globals():
        model_obj = {}
    
    if not model_name in model_obj.keys():
        model = models.get(model_name)
        
    if model:
        model = model()
        model_obj[model_name] = model
        #print(model_name," built")
    else:
        raise ValueError('Invalid model_name passed - {}'.format(model_name))
    return model_obj[model_name]

def represent(img_path, model_name = 'VGG19', model = None, 
              enforce_detection = True, detector_backend = 'opencv', 
              align = True, normalization = 'base'):
    if model is None:
        model = build_model(model_name)
    
    # decide input shape
    input_shape_x, input_shape_y = functions.find_input_shape(model)
    # detect and align
    img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, align = align)
    
    img = functions.normalize_input(img = img, normalization = normalization)
    #represent
    embedding = model.predict(img)[0].tolist()
    
    return embedding
    
    
    
    
    
    