from PIL import Image
import numpy as np
import cv2
import os
import sys

valid_image_endings = ['png', 'jpg', 'jpeg']

def get_hsv_color_ratios():
    # Creates 256 variants of color using HSV
    # 4 bits for color
    # 1 Bit for saturation
    # 3 bit for value
    # 2 values of these bits are designated for white and block
    hue_values = (2**4) - 2 # To allow for black and white colors 
    hue_chunks = [(255*i)//hue_values for i in range(hue_values)]

    saturation_chunks = [100, 255]

    value_values = (2**3)-1
    value_chunks = [50 + (200*i)//value_values for i in range(value_values)]+[255]
    return hue_chunks, saturation_chunks, value_chunks

def select_percentcolor_index(value, chunks, lowest_threshold=0.1):
    # Method for allowing 2 extra colors with saturation and value
    # Due to if any value of saturation and value is 0, it will be either completely black or white
    # Increase then possible color values in get_hsv_ratios, remove these posibility
    # Instead add threshold
    if value < 255*lowest_threshold:
        return len(chunks)
    return min(enumerate(chunks), key=lambda item: abs(item[1] - value))[0]

def select_closest_index(value, chunks):
    return min(enumerate(chunks), key=lambda item: abs(item[1] - value))[0]

def generate_shadow_image(image, width, ratio=None):
    # Generates a binary image of edges at downscaled spatial resolution
    if not ratio:
        ratio = image.shape[0]/image.shape[1]

    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # First detect all edges in original image, with an opening operation
    kernel =cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(grayscaled_image, cv2.MORPH_DILATE, kernel)
    out_gray=cv2.divide(grayscaled_image, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1] 
    result = Image.fromarray(out_binary)

    # Downscales image to correct spatial resolution
    # Then does another threshold to get good binary of image
    image = np.array(result.resize((round(width), round((width)*ratio))))
    # Converts to binary image
    out_binary =  cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
    return out_binary
   
def generate_convert_to_hsv_colorarray(image, width, ratio=None):
    if not ratio:
        ratio = image.shape[0]/image.shape[1]

    # Downscale and convert to HSV
    # HSV is better for comparing colors
    color_image = image.resize((round(width), round((width)*ratio)))
    color_image = np.array(color_image)
    color_image = Image.fromarray(color_image)
    color_image = color_image.convert('HSV')
    return np.array(color_image)

def downscale_color_array(array_image, hue_chunks, saturation_chunks, value_chunks):
    array_image = np.array(array_image, copy=True)
    for x in range(array_image.shape[0]):
        for y in range(array_image.shape[1]):
            array_image[x][y][0] = select_closest_index(array_image[x][y][0], hue_chunks)
            array_image[x][y][1] = select_percentcolor_index(array_image[x][y][1], saturation_chunks)
            array_image[x][y][2] = select_percentcolor_index(array_image[x][y][2], value_chunks)
    return array_image

def save_image_from_array(array_image, filename,  path=""):
    save_image = Image.fromarray(array_image)
    save_image.save(f"{path}binary{filename}".split(".")[0]+".png")

def generate_8bitimage(filename, width=64):
    hue_chunks, saturation_chunks, value_chunks = get_hsv_color_ratios()

    cv2_image = cv2.imread(filename)
    pil_image = Image.open(filename)
    ratio = cv2_image.shape[0]/cv2_image.shape[1]

    out_binary=generate_shadow_image(cv2_image, width, ratio) 
    color_array = generate_convert_to_hsv_colorarray(pil_image, width, ratio)
    color_downscaled_index_array = downscale_color_array(color_array, hue_chunks, saturation_chunks, value_chunks)

    final_image = np.zeros((out_binary.shape[0], out_binary.shape[1], 3))
    for h in range(final_image.shape[0]):
        for w in range(final_image.shape[1]):
            hue = hue_chunks[color_downscaled_index_array[h][w][0]]
            saturation = color_downscaled_index_array[h][w][1]
            if saturation < len(saturation_chunks): 
                saturation = saturation_chunks[saturation]
            else:
                saturation=0
            value = color_downscaled_index_array[h][w][2]
            if value < len(value_chunks): 
                value= value_chunks[max(0, value if out_binary[h][w] == 1 else value-1)]
            else:
                value=0
            final_image[h][w] = (hue, saturation, value)
    save_image = Image.fromarray(final_image.astype(np.uint8), 'HSV').convert('RGB')
    save_image.save(f"results/pixel_{get_filename(filename)}"+".png")

def is_image(filename):
    return os.path.exists(path) and filename.split('.')[-1] in valid_image_endings

def generate_8bitfrom_directory(directory):
    for filename in os.listdir(directory):
        if(is_image(filename)):
            generate_8bitimage(f"{directory}/{filename}")
            print(f"8-bit image generated from {get_filename(filename)}")

def get_filename(path):
    return path.split('/')[-1].split('.')[0]

if __name__=="__main__":
    if len(sys.argv) != 2:
        raise Exception("Requires a file or folder")
    path = sys.argv[1]
    if not os.path.exists(path):
        raise Exception("File or directory not found")
    if is_image(path):
        generate_8bitimage(path)
    else:
        generate_8bitfrom_directory(path)
        print(f"8-bit image generated from {get_filename(path)}")