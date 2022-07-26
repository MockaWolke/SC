import json
import numpy as np
import os
import imageio as io



def get_image_paths(image_id,version="v2.0"):


    image_path = "Data/training/images/{}.jpg".format(image_id)
    label_path = "Data/training/{}/labels/{}.png".format(version, image_id)
    instance_path = "Data/training/{}/instances/{}.png".format(version, image_id)

    return image_path,label_path,instance_path

def get_labels(version="v2.0"):

    with open('Data/config_{}.json'.format(version)) as config_file:
        config = json.load(config_file)

    return config["labels"]

def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array

def get_list_of_image_ids(data="training"):

    path = f"Data/{data}/images"

    return [img_path[:-4] for img_path in os.listdir(path)]


def get_train_image(image_id,mode="original",version="v2.0"):
    """
    Get's images from path, depending on mode
    """

    assert mode in ["original","label","color_label"], 'Possibile modes: "original","label","color_label"'

    if mode== "original":


        path = "Data/training/images/{}.jpg".format(image_id)
        return io.imread(path)

    elif mode=="label":

        path = "Data/training/{}/instances/{}.png".format(version, image_id)
        return np.array(io.imread(path)/"255",dtype=np.uint8)

    elif mode=="color_label":
        
        path = "Data/training/{}/labels/{}.png".format(version, image_id)
        return io.imread(path)[:,:,:3]