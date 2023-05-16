import numpy as np
import pickle
import cv2
from typing import Tuple, List, Union
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def calculate_center_distance(circle1: Tuple[int, int, int], circle2: Tuple[int, int, int]) -> float:
    """
    Calculate the distance between the centers of two circles.
    
    :param circle1: A tuple containing the x, y coordinates and radius of the first circle.
    :param circle2: A tuple containing the x, y coordinates and radius of the second circle.
    :return: The distance between the centers of the two circles.
    """
    return np.sqrt((np.float64(circle1[0]) - np.float64(circle2[0])) ** 2 +
                   (np.float64(circle1[1]) - np.float64(circle2[1])) ** 2)


def update_circle_flags(circle_flags: List[bool], circles: np.ndarray, max_center_distance: float) -> List[bool]:
    """
    Update the circle flags based on the distance between circle centers.
    
    :param circle_flags: A list of boolean flags indicating whether a circle should be kept.
    :param circles: A numpy array containing the circles (x, y, radius).
    :param max_center_distance: The maximum distance between centers of two circles to be considered overlapping.
    :return: The updated list of circle flags.
    """
    for i, circle1 in enumerate(circles[0]):
        for j, circle2 in enumerate(circles[0]):
            if i == j or not circle_flags[i] or not circle_flags[j]:
                continue

            center_distance = calculate_center_distance(circle1, circle2)

            if center_distance <= max_center_distance:
                if circle1[2] > circle2[2]:
                    circle_flags[j] = False
                else:
                    circle_flags[i] = False

    return circle_flags


def filter_inner_circles(circles: np.ndarray, max_center_distance: float) -> np.ndarray:
    """
    Filter out inner circles based on the maximum distance between circle centers.
    
    :param circles: A numpy array containing the circles (x, y, radius).
    :param max_center_distance: The maximum distance between centers of two circles to be considered overlapping.
    :return: A numpy array containing the filtered circles.
    """
    filtered_circles = []
    circle_flags = [True] * len(circles[0])

    circle_flags = update_circle_flags(circle_flags, circles, max_center_distance)

    for i, flag in enumerate(circle_flags):
        if flag:
            filtered_circles.append(circles[0, i])

    return np.array(filtered_circles)


def prepare_img(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare an image by resizing, blurring, and converting to grayscale and RGB.
    
    :param path: The path to the image file.
    :return: A tuple containing the grayscale and RGB versions of the image.
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (1920, 1920))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_gray, img_rgb


def find_circles(img: np.ndarray) -> Union[np.ndarray, None]:
    """
    Find circles in an image using the HoughCircles method.
    
    :param img: A grayscale version of the image.
    :return: A numpy array containing the detected circles, or None if no
    circles are detected.
    """
    return cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 35, param1=300, param2=0.9)


def treat_circles(circles: Union[np.ndarray, None]) -> Union[np.ndarray, None]:
    """
    Round the values of detected circles and convert them to unsigned 16-bit integers.
    
    :param circles: A numpy array containing the detected circles, or None if no circles are detected.
    :return: A numpy array containing the treated circles, or None if no circles are detected.
    """
    if circles is not None:
        return np.uint16(np.around(circles))
    else:
        return None


def draw_circles(img: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """
    Draw circles on the input image.
    
    :param img: The input image in RGB format.
    :param circles: A numpy array containing the circles (x, y, radius) to draw.
    :return: The input image with circles drawn on it
    """
    for circle in circles:
        img = cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 8)
    return img


def extract_coin(image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
    """
    Extract a coin from the input image given its circle (center and radius).

    :param image: The input image in RGB format.
    :param circle: A tuple containing the x, y coordinates and radius of the coin circle.
    :return: A numpy array containing the extracted coin.
    """
    x, y, r = circle
    return image[y - r:y + r, x - r:x + r]


def extract_sift_features(coin: np.ndarray, n_features: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract SIFT features from the given coin image.
    
    :param coin: A numpy array representing the extracted coin image in RGB format.
    :param n_features: The number of desired features to extract (default is 128).
    :return: A tuple containing the keypoint locations and their corresponding descriptors.
    """
    # Convert the RGB coin image to grayscale
    gray_coin = cv2.cvtColor(coin, cv2.COLOR_RGB2GRAY)

    # Initialize the SIFT object
    sift = cv2.SIFT_create(nfeatures=n_features)

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_coin, None)

    return keypoints, descriptors


def create_visual_vocabulary(descriptors_list: List[np.ndarray], n_clusters: int) -> np.ndarray:
    """
    Create a visual vocabulary using k-means clustering on the SIFT descriptors.
    
    :param descriptors_list: A list of numpy arrays containing SIFT descriptors for each coin.
    :param n_clusters: The number of clusters (visual words) to be created.
    :return: A numpy array containing the cluster centers (visual words).
    """
    all_descriptors = np.vstack(descriptors_list)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(all_descriptors)
    
    return kmeans.cluster_centers_


def calculate_histogram_of_visual_words(descriptors: np.ndarray, visual_words: np.ndarray) -> np.ndarray:
    """
    Calculate the histogram of visual words for a given set of SIFT descriptors.
    
    :param descriptors: A numpy array containing the SIFT descriptors for a single coin.
    :param visual_words: A numpy array containing the visual words (cluster centers).
    :return: A numpy array representing the histogram of visual words.
    """
    closest_visual_words, _ = pairwise_distances_argmin_min(descriptors, visual_words)
    histogram = np.bincount(closest_visual_words, minlength=len(visual_words))
    
    return histogram


def create_histogram_of_visual_words(image: np.array, circles: List[Tuple[int, int, int]]) -> List[np.array]:
    """
    Process the given image and extract the histogram of visual words for each coin in the image.

    Args:
    img_rgb (np.array): The input image in RGB format.
    circles (list): A list of circles detected in the image. Each circle is represented by a tuple (x, y, radius).

    Returns:
    list of np.array: A list of histograms of visual words. Each histogram corresponds to a coin in the image.
    """
    
    # Get only the coins from the image
    coins = [extract_coin(image, circle) for circle in circles]
        
    # Apply SIFT to the coins
    sift_results = [extract_sift_features(coin, 128) for coin in coins]
    descriptors = [result[1] for result in sift_results]
        
    # Create visual vocabulary
    visual_vocabulary = create_visual_vocabulary(descriptors, 100)
        
    # Get coins histogram of visual words
    coins_histograms = [calculate_histogram_of_visual_words(descriptor, visual_vocabulary) for descriptor in descriptors]
    
    return coins_histograms


def update_training_data(coins_histograms: List[np.array], labels: List[str], training_data: List[Tuple[np.array, str]]) -> None:
    """
    Update training data by appending each coin's histogram of visual words along with its corresponding label.

    Args:
    coins_histograms (list of np.array): A list of histograms of visual words. Each histogram corresponds to a coin.
    labels (list of str): A list of labels. Each label corresponds to a coin.
    training_data (list of tuple): The existing training data list which will be updated.

    Returns:
    None. The training_data list passed as an argument will be updated.
    """
    for histogram, label in zip(coins_histograms, labels):
        training_data.append((histogram, label))


def save_training_data(training_data: List[Tuple[np.array, str]], filename: str) -> None:
    """
    Save the training data to a file.

    Args:
    training_data (list of tuple): The training data to save.
    filename (str): The name of the file to save the training data to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(training_data, f)
        
        
def calculate_total_value(predictions: List[str]) -> float:
    """
    Calculate the total value of coins detected in the image, in Brazilian Real.

    Args:
    predictions (List[str]): A list of predicted coin labels.

    Returns:
    float: The total value of coins in the image, in Brazilian Real.
    """
    coin_values = {
        '1 real': 1.00,
        '50 centavos': 0.50,
        '25 centavos': 0.25,
        '10 centavos': 0.10,
        '5 centavos': 0.05
    }

    total_value = 0
    for prediction in predictions:
        if prediction in coin_values:
            total_value += coin_values[prediction]
    
    return round(total_value, 2)


def write_to_log(log_filename: str, image_number: int, total_value: float):
    """
    Write the total value of coins in an image to a log file.
    
    Args:
    log_filename (str): The name of the log file.
    image_number (int): The number of the image.
    total_value (float): The total value of coins in the image.
    
    Returns:
    None
    """
    with open(log_filename, 'a') as f:
        f.write(f"Image {image_number+1}: {total_value:.2f} reais, 0 dólares, 0 euros, 0 libras, 0 pesos, 0 bitcoins\n")
