import utils
import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

paths = utils.get_list_of_image_ids()

ds = tf.data.Dataset.from_tensor_slices(paths)


def mapping_func_train_ogimages(image_path):
    """
    Function to be mapped on tfds to read+downsize images.
    -----------------
    arguments:
    image_path - filename of specific image
    -----------------
    returns:
    Resized image and its filename.
    """

    image = tf.io.read_file("Data/training/images/" + image_path + ".jpg")
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224,224])
    return image, image_path

def mapping_func_train_instances(image_path):
    """
    Function to be mapped on tfds to read+downsize images.
    -----------------
    arguments:
    image_path - filename of specific image
    -----------------
    returns:
    Resized image and its filename.
    """

    image = tf.io.read_file("Data/training/v2.0/instances/" + image_path + ".png")
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [448,448])
    return image, image_path







ds = ds.map(mapping_func_train_instances)

# save every image to the path
for img, path in tqdm.tqdm(tfds.as_numpy(ds)):
    tf.keras.utils.save_img("Resized_Data/training/instances/" + path.decode("utf-8") + ".jpg", img)