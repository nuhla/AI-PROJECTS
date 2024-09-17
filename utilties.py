from PIL import Image ,UnidentifiedImageError
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import os
import json as json

# ------------------------------------------------- # 
#-----------------normlize image -------------------#
#---------------------------------------------------#
def process_image(image):


    image =tf.cast(image, tf.float32)
    image = tf.image.resize(image,[224,224],method='nearest')
    image/=255

    return image.numpy()


#     #---------------------------------------------------#
#     #------------------Process Image -------------------#
#     #---------------------------------------------------# 
def predict(image_path,model,topk=5):


    imge = Image.open(image_path)
    imge = process_image(imge)
    new_image_batch = np.stack((imge,), axis=0)

#     #---------------------Load Model -------------------#

    KeraseHublayer=tf.keras.models.load_model("./models/"+model,  compile = True, custom_objects={'KerasLayer':hub.KerasLayer}) 

#     #------------------Predict Image -------------------#

    outputs =KeraseHublayer.predict(new_image_batch)
    top_k_values, top_k_indices = tf.nn.top_k(list(tf.cast(outputs, tf.float32)), k=topk)
    return  top_k_values.numpy()[0], top_k_indices[0].numpy()



#     #---------------------------------------------------#
#     #------------------chek Image Path -----------------#
#     #---------------------------------------------------#
def chek_image_path(path):      
    try :
        Image.open(path)
        return True
    except (UnidentifiedImageError, IOError):
        return False


#     #---------------------------------------------------#
#     #------------------chek Model Path -----------------#
#     #---------------------------------------------------#
def chek_model_path(model_name):    
       return os.path.isfile("./models/"+model_name)



#     #---------------------------------------------------#
#     #----------- Read json and return values ------------#
#     #---------------------------------------------------#
def red_json_file(path, array=np.array([])):
    with open(path, 'r') as f:
        class_names = json.load(f)

    finale_values = []
    if len(array)!=0:
          for item in np.array(array):
              print(item)
              finale_values.append(class_names[str(item)])

    return finale_values