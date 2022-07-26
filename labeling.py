from utils import *
import matplotlib.pyplot as plt
import json


"""
This is for creating manual labels
"""

#label_dir = json.load(open('self_made_labels/labels.json',"r"))

label_dir = {}


i = int(open('self_made_labels/img_count.txt',"r").read())

print(f"We start with image {i}")


train_image_ids = get_list_of_image_ids()

print("Labels: a - Image of Street, j - image of sidewalk,n - not clear, b - last image -> change")

while True:

    img_id = train_image_ids[i]

    plt.imshow(get_train_image(img_id,"original"))
    plt.axis("off")
    plt.show()
    
    val = ""
    
    while val not in ["a","j","b","n"]:
        val = input("Which_label? ")

    if val == "b":
        i -=1

    else:

        if val=="a":
            label_dir[img_id] = 1
        elif val == "j":
            label_dir[img_id] = 2
        elif val == "n":
            label_dir[img_id] = 3


        with open('self_made_labels/labels.json', 'w') as fp:
            json.dump(label_dir, fp)

        with open('self_made_labels/img_count.txt',"w") as f:

            f.write(str(i))

        i+=1