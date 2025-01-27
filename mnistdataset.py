import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten the images and combine train and test data
train_data = train_images.reshape(train_images.shape[0], -1)
test_data = test_images.reshape(test_images.shape[0], -1)

# Create DataFrame for train and test data
train_df = pd.DataFrame(train_data)
train_df.insert(0, "Label", train_labels)  # Insert labels as the first column

test_df = pd.DataFrame(test_data)
test_df.insert(0, "Label", test_labels)  # Insert labels as the first column

# Combine train and test DataFrames
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Save to a single Excel file
output_file = "mnist_all_data.xlsx"
combined_df.to_excel(output_file, index=False, sheet_name="MNIST Data", engine="openpyxl")

print(f"All data saved successfully to {output_file}")
