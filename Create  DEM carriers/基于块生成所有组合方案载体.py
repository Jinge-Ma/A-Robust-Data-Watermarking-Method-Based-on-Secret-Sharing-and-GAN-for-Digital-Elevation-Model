from itertools import combinations
from PIL import Image
import numpy as np


# Function to perform XOR operation on a list of images
def xor_images(image_list):
    """
    Perform XOR operation on a list of images.
    :param image_list: List of Image objects to XOR.
    :return: Resultant Image object after XOR operation.
    """
    # Start with the first image's array
    result_array = np.array(image_list[0])
    # Perform XOR with the rest of the images
    for img in image_list[1:]:
        img_array = np.array(img)
        result_array = np.bitwise_xor(result_array, img_array)
    # Convert the resulting array back to an image
    return Image.fromarray(result_array)


# Paths to the uploaded images
image_paths = [
    './等值线/binary_contour_block_0_0.png',
    './等值线/binary_contour_block_0_1.png',
    './等值线/binary_contour_block_0_2.png',
    './等值线/binary_contour_block_1_0.png',
    './等值线/binary_contour_block_1_1.png',
    './等值线/binary_contour_block_1_2.png',
    './等值线/binary_contour_block_2_0.png',
    './等值线/binary_contour_block_2_1.png',
    './等值线/binary_contour_block_2_2.png'
]

# Load images from the paths
loaded_images = [Image.open(path).convert('1') for path in image_paths]

# Generate all possible non-empty combinations of image indices
n = len(loaded_images)
all_combinations = [combo for r in range(1, n + 1) for combo in combinations(range(n), r)]

# Perform XOR on all combinations and save the resultant images
for i, combination in enumerate(all_combinations, 1):
    # Get the actual images for the current combination
    images_to_xor = [loaded_images[idx] for idx in combination]

    # Perform XOR operation on the images
    result_image = xor_images(images_to_xor)


    # Optional: Convert the image to 'P' mode for more efficient PNG compression
    result_image = result_image.convert('P')

    # Save the result to a file with optimization
    final_filename = f'./binary_images/{i}.png'
    print(f'Saving {final_filename}...')
    result_image.save(final_filename, format='PNG', optimize=True)

    print(i,combination)
