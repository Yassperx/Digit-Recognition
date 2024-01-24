from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()

# Save all images
for i, image in enumerate(digits.images):
    plt.imsave(f'digits/digit#{i}.png', image, cmap='gray')