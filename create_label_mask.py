import numpy as np
import matplotlib.pyplot as plt
import utils
import json

labels = utils.get_labels()
train_image_ids = utils.get_list_of_image_ids()

# found_in_image = {}

found_in_image = json.load(open('example_images_labels/label_examples.json', 'r'))
todo = [(a,b["name"]) for a,b in enumerate(labels) if str(a) not in found_in_image]

def dump():

    with open('example_images_labels/label_examples.json', 'w') as fp:
        json.dump(found_in_image, fp)

def create_label_img(i):


    img_id = found_in_image[str(i)][1]
    per = found_in_image[str(i)][2]

    label_img = utils.get_train_image(img_id,"label")


    original_img = utils.get_train_image(img_id,"original")
    
    copy = original_img.copy()
    
    original_img[label_img==i] = original_img[label_img==i][:,2]=255

    mod = np.zeros(label_img.shape)
    mod[label_img==i] = 255

    fig, ax = plt.subplots(1,3,figsize=(10,5))

    ax[0].axis('off')
    ax[0].imshow(original_img)
    ax[1].axis('off')
    ax[2].axis('off')

    ax[1].imshow(mod)

    ax[2].imshow(copy)

    name = labels[i]["readable"]
    fig.suptitle(f"{name}, {per:.2f}% of img")

    plt.tight_layout()
    plt.savefig(f"example_images_labels/{name}.jpg")

if __name__ == "__main__":

    i = 400

    while todo:

        img = utils.get_train_image(train_image_ids[i],"label")

        to_del = []

        for val,name in todo:

            count = (img==val).sum()

            percentag_of_img = count / np.product(img.shape)

            if i< 600:
                bool = percentag_of_img > 0.015
            else: 
                bool = percentag_of_img > 0.002

            if bool:

                n_left = int(len(todo)-len(to_del)-1)

                print(f"Found {name} in image number {i}, {n_left} left")
                if n_left <= 10:

                    print(todo)
                
                found_in_image[str(val)] = (name,train_image_ids[i],percentag_of_img)
                create_label_img(val)
                dump()

                to_del.append((val,name))

        for label in to_del:

            todo.remove(label)


        i+=1

        if i%50 == 0:
            print(i)


