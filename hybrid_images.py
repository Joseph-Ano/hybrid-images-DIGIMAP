import typing as T
import numpy as np
import cv2
import imageio

# Do not import additional modules, otherwise, there will be deductions

def normalize_img(image: str) -> np.ndarray:
    image_data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(image_data)
    return image_array / 255 

def add_padding(image: np.ndarray, kernel_size: int, maxH: int, maxW: int) -> np.ndarray:
    padding = kernel_size//2
    newH = maxH - image.shape[0] + padding
    newW = maxW - image.shape[1] + padding
    padded_img = cv2.copyMakeBorder(image, padding, newH, padding, newW, cv2.BORDER_CONSTANT, None, value=0)
    return padded_img

def hybrid_images(image_high: T.Union[str, np.ndarray], image_low: T.Union[str, np.ndarray], output_file: str = None) -> np.ndarray:
    """
    Creates a hybrid image by combining a high-pass filtered version of the first input image
    with a low-pass filtered version of the second input image. The resulting image is a
    mixture of the high-frequency content of the first image and the low-frequency content of
    the second image.

    Args:
        image_high (Union[str, np.ndarray]): The first input image, either a filename (str) or a numpy array of shape CxHxW.
        image_low (Union[str, np.ndarray]): The second input image, either a filename (str) or a numpy array of shape CxHxW.
        output_file (str, optional): The filename to save the resulting image if not None.

    Returns:
        np.ndarray: The resulting hybrid image, as a numpy array (always returns a value)
    """

    KERNEL_SIZE = 13

    if(output_file == None):
        output_file = "hybrid_img.png"
        
    if(type(image_high) == str):
        normalized_image_high = normalize_img(image_high)
    else:
        cxhxw_2_hxwxc = np.transpose(image_high, (1, 2, 0))
        rbg_2_gray = cv2.cvtColor(cxhxw_2_hxwxc, cv2.COLOR_RGB2GRAY)
        normalized_image_high = rbg_2_gray / 255

    if(type(image_low) == str):
        normalized_image_low = normalize_img(image_low)
    else:
        cxhxw_2_hxwxc = np.transpose(image_low, (1, 2, 0))
        rbg_2_gray = cv2.cvtColor(cxhxw_2_hxwxc, cv2.COLOR_RGB2GRAY)
        normalized_image_low = rbg_2_gray / 255

    # Manually align the two images
    normalized_image_low =  cv2.copyMakeBorder(normalized_image_low, 19, 0, 22, 0, cv2.BORDER_CONSTANT, None, value=0)

    maxH = max(normalized_image_high.shape[0], normalized_image_low.shape[0])
    maxW = max(normalized_image_high.shape[1], normalized_image_low.shape[1])

    normalized_image_high = add_padding(normalized_image_high, KERNEL_SIZE, maxH, maxW)
    normalized_image_low = add_padding(normalized_image_low, KERNEL_SIZE, maxH, maxW)

    low_freq = cv2.GaussianBlur(normalized_image_low, (KERNEL_SIZE,KERNEL_SIZE), 5)
    high_freq = normalized_image_high - cv2.GaussianBlur(normalized_image_high, (KERNEL_SIZE,KERNEL_SIZE), 4)

    hybrid_img = np.uint8((low_freq + high_freq) * 255)

    cropped_hybrid_img = hybrid_img[30:550, 30:450]

    cv2.imwrite(output_file, cropped_hybrid_img)

    return cropped_hybrid_img

if __name__ == "__main__":

    image = hybrid_images("high_img.png", "low_img.png", "hybrid_img.png")  # Find images on your own
    print(image.shape)
