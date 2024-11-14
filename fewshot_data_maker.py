import os

def delete_images_except_first_16(folder_path):
    # Loop through each subfolder in the main folder
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        
        # Check if it is indeed a directory
        if os.path.isdir(subfolder_path):
            # Get a list of all files in the subfolder, sorted by name
            files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))])
            
            # Delete all images except the first 16
            for file_name in files[16:]:
                file_path = os.path.join(subfolder_path, file_name)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Example usage
delete_images_except_first_16('DataDream/data/eurosat/real_train_fewshot/seed0')
