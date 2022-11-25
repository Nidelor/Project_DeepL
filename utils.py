#camila
import os
from tqdm import tqdm
#Preprocessing audio folder
import numpy as np 
#add relative

    
def vector_to_scenes(scenes_vector):
    f = open("sound/categories_places2.txt",'r')
    categories = f.read().split("\n")
    for example in range((scenes_vector.shape[0])):
        for x in range((scenes_vector.shape[2])):
            index = np.argmax(scenes_vector[example,:,x,0])
            print(categories[index])

def vector_to_obj(obj):
    f = open("sound/objects.txt",'r')
    categories = f.read().split("\n")
    for example in range(obj.shape[0]):
        for x in range(obj.shape[2]):
            index = np.argmax(obj[example,:,x,0])
            print(categories[index])
