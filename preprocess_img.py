import os
from PIL import Image


def process_images_in_directory(directory='/home/jz927/Documents/relighting/neuralgaffer/Neural_Gaffer/preprocessed_data7/img'):
    # Browse the current folder for all PNG files
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)

            # Open the image
            image = Image.open(file_path)

            # Check if the image is in RGBA mode
            if image.mode == 'RGBA':
                # Create a white background image
                white_bg = Image.new('RGB', image.size, (255, 255, 255))
                # Paste the RGBA image on top of the white background
                white_bg.paste(image, (0, 0), image)
                image = white_bg  # Update image to the new RGB image with a white background

            # Check if the image is 256x256, if not, resize it
            if image.size != (256, 256):
                image = image.resize((256, 256))

            # Save the processed image with the same name
            image.save(file_path)
            print(f"Processed and saved: {filename}")


# Example usage
process_images_in_directory()  # This will process images in the current directory
