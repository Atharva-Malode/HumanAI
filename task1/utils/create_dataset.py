import os
import shutil
import pandas as pd

class CreateDataset:
    """
    Class to create a new dataset by randomly selecting images from each class.
    The selected images are moved to a new directory inside the dataset folder, organized by the class labels.
    Additionally, a CSV file is created to store the new dataset's image paths and labels.
    """
    def __init__(self, csv_file, root_directory, new_dataset_directory, image_col, class_col, buffer_size=300, num_classes=24, name="dataset"):
        self.data = pd.read_csv(csv_file)
        
        # Remove leading/trailing spaces from column names
        self.data.columns = self.data.columns.str.strip()
        
        # Ensure the provided column names are correct after stripping spaces
        self.image_col = image_col.strip()  
        self.class_col = class_col.strip()  
        
        self.root_directory = root_directory
        self.new_dataset_directory = new_dataset_directory
        self.buffer_size = buffer_size
        self.num_classes = num_classes  
        self.name = name  
    
    def create(self):
        """
        Create a new dataset by randomly selecting images from each class.
        The selected images are moved to a new dataset directory under a specified subfolder name.
        Additionally, a CSV file is created to store the new dataset's image paths and labels.
        """
        dataset_folder = os.path.join(self.new_dataset_directory, self.name)
        os.makedirs(dataset_folder, exist_ok=True)
        
        csv_folder = os.path.join(self.new_dataset_directory, "csv")
        os.makedirs(csv_folder, exist_ok=True)
        
        csv_file_path = os.path.join(csv_folder, f"{self.name}.csv")
        
        records = []  
        
        for class_label in range(self.num_classes):
            class_folder = os.path.join(dataset_folder, str(class_label))
            os.makedirs(class_folder, exist_ok=True)
        
        for class_label in range(self.num_classes):  
            class_data = self.data[self.data[self.class_col] == class_label]

            if class_data.empty:
                print(f"Warning: Class {class_label} has no images.")
                continue  

            if len(class_data) < self.buffer_size:
                random_sample = class_data
                print(f"Class {class_label} has fewer than {self.buffer_size} images. Taking all {len(class_data)} images.")
            else:
                random_sample = class_data.sample(n=self.buffer_size, random_state=42)

            for _, row in random_sample.iterrows():
                image_path = row[self.image_col]  
                image_name = os.path.basename(image_path)
                
                source_path = os.path.join(self.root_directory, image_path)
                class_folder = os.path.join(dataset_folder, str(class_label))
                destination_path = os.path.join(class_folder, image_name)
                
                print(f"Source path: {source_path}")
                print(f"Destination path: {destination_path}")
                
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                    records.append([destination_path, class_label])  
                else:
                    print(f"Warning: Image {image_name} does not exist at {source_path}")
        
        new_df = pd.DataFrame(records, columns=["image_path", "class_label"])
        new_df.to_csv(csv_file_path, index=False)
        print(f"Dataset created and images moved to {dataset_folder}")
        print(f"CSV file saved at {csv_file_path}")
