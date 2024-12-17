import os
from collections import defaultdict
import json

def create_image_label_dict(root_dir):
    # Define classes in alphabetical order
    classes = sorted([
        'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 
        'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 
        'Ulcer', 'Worms'
    ])
    
    # Dictionary to store image filename and corresponding label list
    image_dict = defaultdict(lambda: [0] * 10)  # Initialize labels with a list of 10 zeroes
    
    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(root_dir, class_name)
        
        # Iterate through image files in the class folder
        for filename in os.listdir(class_path):
            if filename.endswith('jpg') or filename.endswith('png'):
                file_path = os.path.join(class_path, filename)
            
                if os.path.isfile(file_path):
                    # Mark the class position as 1 for the corresponding image
                    image_dict[filename][class_index] = 1

    return image_dict

# Usage
train_dir = 'Dataset/training'  # Path to training
valid_dir = 'Dataset/validation'  # Path to validation
train_label_dict = create_image_label_dict(train_dir)
valid_label_dict = create_image_label_dict(valid_dir)

# Print dictionary and count
c=0
for image, labels in train_label_dict.items():
    if sum(labels)>1:
        c+=1
        print(image, end=" ")
        if sum(labels)>2:
            print(sum(labels), end=" ")
        print()
print(c)
c=0
for image, labels in valid_label_dict.items():
    if sum(labels)>1:
        c+=1
        print(image, end=" ")
        if sum(labels)>2:
            print(sum(labels), end=" ")
        print()
print(c)
print(len(train_label_dict))
print(len(valid_label_dict))
# Ignore .DS_Store since its a mac thing
# print(".DS_Store: ", image_label_dict['.DS_Store'])
# del image_label_dict['.DS_Store']
# for image in image_label_dict:
#     if image==".DS_Store":
#         print("yo")

# Convert and write JSON object to file
with open("train_labels.json", "w") as outfile: 
    json.dump(train_label_dict, outfile)

with open("valid_labels.json", "w") as outfile: 
    json.dump(valid_label_dict, outfile)