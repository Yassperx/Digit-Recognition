from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

digits = load_digits()

# Split the data into training and test sets (70% training, 30% test)
def make_sets(digits, tst_size=0.3)
    X_train, X_test, _, _ = train_test_split(
        digits.images, digits.target, test_size=tst_size, random_state=42)
    return X_train, X_test

# Create 'digits/training' and 'digits/test' directories if they don't exist
def make_dataset(X_train, X_test):
    training_directory = 'digits/training'
    test_directory = 'digits/test'
    os.makedirs(training_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    # Save images to training set with tqdm progress bar
    for i, image in tqdm(enumerate(X_train), total=len(X_train), desc='Training set'):
        plt.imsave(os.path.join(training_directory, f'digit#{i}.png'), image, cmap='gray')

    # Save images to test set with tqdm progress bar
    for i, image in tqdm(enumerate(X_test), total=len(X_test), desc='Test set'):
        plt.imsave(os.path.join(test_directory, f'digit#{i}.png'), image, cmap='gray')