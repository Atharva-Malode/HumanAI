import os
import shutil
import pandas as pd

class CreateEvalDataset:
    def __init__(self, root_directory, eval_data_directory, min_samples_per_class=20, 
                 max_total_samples=500, folder_name="eval_data"):
        self.root_directory = root_directory
        self.eval_data_directory = eval_data_directory
        self.folder_name = folder_name
        self.min_samples_per_class = min_samples_per_class
        self.max_total_samples = max_total_samples

    def create(self, csv_file, image_col, class_col):
        data = pd.read_csv(csv_file, usecols=[image_col, class_col])
        data[class_col] = data[class_col].astype(str) 

        eval_data_folder = os.path.join(self.eval_data_directory, self.folder_name)
        os.makedirs(eval_data_folder, exist_ok=True)

        csv_folder = os.path.join(self.eval_data_directory, "csv")
        os.makedirs(csv_folder, exist_ok=True)
        csv_file_path = os.path.join(csv_folder, f"{self.folder_name}.csv")

        records = []
        total_samples_collected = 0

        unique_classes = data[class_col].unique()

        for class_label in unique_classes:
            class_data = data[data[class_col] == class_label]
            num_samples = len(class_data)
            
            if num_samples < self.min_samples_per_class:
                print(f"Warning: Class {class_label} has only {num_samples} images. Taking all.")
                selected_data = class_data
            else:
                selected_data = class_data.sample(n=self.min_samples_per_class, random_state=42)

            for _, row in selected_data.iterrows():
                image_path = row[image_col]
                image_name = os.path.basename(image_path)
                source_path = os.path.join(self.root_directory, image_path)
                destination_path = os.path.join(eval_data_folder, image_name)

                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                    records.append([destination_path, class_label])
                    total_samples_collected += 1
                else:
                    print(f"Warning: Image {image_name} does not exist at {source_path}")

                if total_samples_collected >= self.max_total_samples:
                    break

            if total_samples_collected >= self.max_total_samples:
                break

        pd.DataFrame(records, columns=["image_path", "class_label"]).to_csv(csv_file_path, index=False)
        print(f"Dataset created at {eval_data_folder}")
        print(f"CSV saved at {csv_file_path}")