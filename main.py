# #!/usr/bin/env python
# # coding: utf-8

# import os
# import numpy as np
# import cv2
# from skimage.feature import hog
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score
# import tkinter as tk
# from tkinter import filedialog, messagebox

# class MaskDetectionApp:
#     def __init__(self, master):
#         self.master = master
#         master.title("Mask Detection")

#         self.label = tk.Label(master, text="Enter the path to the image you want to predict:")
#         self.label.pack()

#         self.input_path_entry = tk.Entry(master)
#         self.input_path_entry.pack()

#         self.browse_button = tk.Button(master, text="Browse", command=self.browse_image)
#         self.browse_button.pack()

#         self.predict_button = tk.Button(master, text="Predict", command=self.predict_image)
#         self.predict_button.pack()

#         self.quit_button = tk.Button(master, text="Quit", command=master.quit)
#         self.quit_button.pack()

#     def browse_image(self):
#         filename = filedialog.askopenfilename(initialdir="/", title="Select file",
#                                               filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
#         self.input_path_entry.delete(0, tk.END)
#         self.input_path_entry.insert(0, filename)

#     def predict_image(self):
#         input_image_path = self.input_path_entry.get()
#         if not input_image_path:
#             messagebox.showerror("Error", "Please select an image.")
#             return

#         try:
#             input_image = cv2.imread(input_image_path)
#             if input_image is not None:
#                 input_image_resized = cv2.resize(input_image, (128, 128))
#                 input_features = extract_features([input_image_resized])

#                 # Making prediction
#                 prediction = logistic_model.predict(input_features)

#                 if prediction[0] == 1:
#                     messagebox.showinfo("Result", "The person in the image is wearing a mask")
#                 else:
#                     messagebox.showinfo("Result", "The person in the image is not wearing a mask")
#             else:
#                 messagebox.showerror("Error", "Invalid image. Please select a valid image file.")
#         except Exception as e:
#             messagebox.showerror("Error", f"An error occurred: {e}")

# def load_dataset(data_directory):
#     X = []
#     y = []

#     for class_name in os.listdir(data_directory):
#         class_directory = os.path.join(data_directory, class_name)
#         if os.path.isdir(class_directory):
#             # Assign class labels based on folder names
#             if class_name == 'with_mask':
#                 class_label = 1  # Class label for 'with_mask'
#             elif class_name == 'without_mask':
#                 class_label = 0  # Class label for 'without_mask'
#             else:
#                 continue  # Skip folders that are not 'with_mask' or 'without_mask'

#             for filename in os.listdir(class_directory):
#                 img_path = os.path.join(class_directory, filename)
#                 img = cv2.imread(img_path)
#                 img_resized = cv2.resize(img, (128, 128))
#                 X.append(img_resized)
#                 y.append(class_label)

#     return np.array(X), np.array(y)

# def extract_features(images):
#     features = []
#     for img in images:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
#                            cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
#         features.append(hog_features)
#     return np.array(features)

# def main():
#     # Path to the downloaded dataset
#     data_directory = 'dataset/'

#     # Load the dataset
#     X, y = load_dataset(data_directory)

#     # Check if there are at least two classes present
#     unique_classes = np.unique(y)
#     if len(unique_classes) < 2:
#         print("Error: The dataset must contain at least two classes for logistic regression classification.")
#         return

#     # Split the dataset into training and validation sets
#     X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Extract HOG features from the images
#     X_train_features = extract_features(X_train)

#     # Create and train the logistic regression model
#     global logistic_model
#     logistic_model = LogisticRegression(max_iter=1000)
#     logistic_model.fit(X_train_features, y_train)

#     # GUI
#     root = tk.Tk()
#     app = MaskDetectionApp(root)
#     root.mainloop()

# if __name__ == '__main__':
#     main()




import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox
from joblib import dump, load

class MaskDetectionApp:
    def __init__(self, master, model_path, X_val_features, y_val):
        self.master = master
        master.title("Mask Detection")

        self.label = Label(master, text="Enter the path to the image you want to predict:")
        self.label.pack()

        self.input_path_entry = Entry(master)
        self.input_path_entry.pack()

        self.browse_button = Button(master, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict_image)
        self.predict_button.pack()

        self.quit_button = Button(master, text="Quit", command=master.quit)
        self.quit_button.pack()

        # Load the model
        self.model = load(model_path)

        # Display accuracy
        self.accuracy_label = Label(master, text=f"Model Accuracy: {self.model.score(X_val_features, y_val)}")
        self.accuracy_label.pack()

    def browse_image(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                              filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.input_path_entry.delete(0, "end")
        self.input_path_entry.insert(0, filename)

    def predict_image(self):
        input_image_path = self.input_path_entry.get()
        if not input_image_path:
            messagebox.showerror("Error", "Please select an image.")
            return

        try:
            input_image = cv2.imread(input_image_path)
            if input_image is not None:
                input_image_resized = cv2.resize(input_image, (128, 128))
                input_features = extract_features([input_image_resized])

                # Making prediction
                prediction = self.model.predict(input_features)

                if prediction[0] == 1:
                    messagebox.showinfo("Result", "The person in the image is wearing a mask")
                else:
                    messagebox.showinfo("Result", "The person in the image is not wearing a mask")
            else:
                messagebox.showerror("Error", "Invalid image. Please select a valid image file.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

def load_dataset(data_directory):
    X = []
    y = []

    for class_name in os.listdir(data_directory):
        class_directory = os.path.join(data_directory, class_name)
        if os.path.isdir(class_directory):
            # Assign class labels based on folder names
            if class_name == 'with_mask':
                class_label = 1  # Class label for 'with_mask'
            elif class_name == 'without_mask':
                class_label = 0  # Class label for 'without_mask'
            else:
                continue  # Skip folders that are not 'with_mask' or 'without_mask'

            for filename in os.listdir(class_directory):
                img_path = os.path.join(class_directory, filename)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (128, 128))
                X.append(img_resized)
                y.append(class_label)

    return np.array(X), np.array(y)

def extract_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
        features.append(hog_features)
    return np.array(features)

def main():
    # Path to the downloaded dataset
    data_directory = 'dataset/'

    # Load the dataset
    X, y = load_dataset(data_directory)

    # Check if there are at least two classes present
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("Error: The dataset must contain at least two classes for logistic regression classification.")
        return

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract HOG features from the images
    X_train_features = extract_features(X_train)
    X_val_features = extract_features(X_val)

    # Create and train the logistic regression model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_features, y_train)

    # Save the model
    dump(logistic_model, "mask_detection_model.joblib")

    # GUI
    root = Tk()
    app = MaskDetectionApp(root, "mask_detection_model.joblib", X_val_features, y_val)
    root.mainloop()

if __name__ == '__main__':
    main()
