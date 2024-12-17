import os
import shutil

def move_images_to_class_folders(root_dir):
    for split in ['training', 'validation']:  # Loop through both 'training' and 'validation'
        split_path = os.path.join(root_dir, split)
        
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            
            if os.path.isdir(class_path):
                # Iterate over subfolders (like KVASIR, SEE-AI, etc.)
                for source_name in os.listdir(class_path):
                    source_path = os.path.join(class_path, source_name)
                    
                    if os.path.isdir(source_path):
                        # Move all files from the source folder to the class folder
                        for filename in os.listdir(source_path):
                            file_path = os.path.join(source_path, filename)
                            
                            if os.path.isfile(file_path):
                                # Move the file to the parent class folder
                                shutil.move(file_path, class_path)
                        
                        # After moving the files, remove the empty source directory
                        os.rmdir(source_path)

# Specify the root directory where the 'training' and 'validation' folders exist
root_dir = '../Dataset'
move_images_to_class_folders(root_dir)
