"""
This program performs data augmentation for segmentation task where each annotation is a 4-point polygon.

Usage:
    python augment.py --data config.yaml --no-show
    python augment.py --data config.yaml --show
"""


import cv2
import pandas as pd
import random
import numpy as np
import argparse
from pathlib import Path 
import yaml 
from tqdm import tqdm



def load(img_path, label_path):
    """
    Loads an image and its corresponding labels from specified file paths.

    Args:
        img_path (str): The file path to the image.
        label_path (str): The file path to the labels.

    Returns:
        tuple: A tuple containing the loaded image (as a BGR NumPy array) and the labels.
               Returns None if there is an issue loading the image.

    Example:
        >>> img, labels = load("path/to/image.jpg", "path/to/labels.txt")
    """
    bgr_img = cv2.imread(img_path)
    labels = load_labels(label_path)
    return (bgr_img, labels)



def load_labels(labels_txt_file_path):
    """
    Loads labels from a text file into a Pandas DataFrame.

    The text file is expected to be whitespace-separated, with columns representing
    label class and bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4).

    Args:
        labels_txt_file_path (str): The file path to the labels text file.

    Returns:
        pandas.DataFrame: A DataFrame containing the loaded labels.
                          Returns None if there is an issue loading the file.

    Example:
        Assuming a labels.txt file with the following content:
        0 10 20 30 40 50 60 70 80
        1 100 200 300 400 500 600 700 800

        >>> df = load_labels("labels.txt")
        >>> print(df)
           label    x1    y1    x2    y2    x3    y3    x4    y4
        0      0  10.0  20.0  30.0  40.0  50.0  60.0  70.0  80.0
        1      1 100.0 200.0 300.0 400.0 500.0 600.0 700.0 800.0
    """
    column_names = ['label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']    
    dtype_mapping = {col: np.float32 for col in column_names[1:]}  # All except 'label' as float
    dtype_mapping['label'] = np.int32  # 'label' as int
    df = pd.read_csv(labels_txt_file_path, sep=r'\s+', names=column_names, dtype=dtype_mapping)
    return df



def preprocess(img, labels, args):
    """
    Preprocesses an image and its labels by resizing the image and filtering labels.

    Args:
        img (numpy.ndarray): The input image (NumPy array).
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
        args (argparse.Namespace or similar): An object containing preprocessing arguments,
                                               including 'img_size' (tuple: height, width) and
                                               'classes' (list: label classes to keep).

    Returns:
        tuple: A tuple containing the resized image (NumPy array) and the filtered and rearranged labels
               (Pandas DataFrame).
    """
    h, w = args.img_size
    new_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    new_labels = labels[labels['label'].isin(args.classes)]
    new_labels = rearrange(new_labels)
    return (new_img, new_labels)



def rearrange(labels):
    """
    Rearranges the coordinates of bounding box polygons within a Pandas DataFrame.

    This function reshapes the x and y coordinates from the input DataFrame into
    a polygon representation (4 points, each with x and y), applies a transformation
    using the 'get_polygon' function (assumed to be defined elsewhere), and then
    updates the original DataFrame with the rearranged coordinates.

    Args:
        labels (pandas.DataFrame): A DataFrame containing bounding box coordinates.
                                     Expected columns: 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'.

    Returns:
        pandas.DataFrame: The input DataFrame with the bounding box coordinates rearranged.
    """
    polygons = np.array(labels[['x1','y1','x2','y2','x3','y3','x4','y4']])
    polygons = polygons.reshape(-1, 4, 2)
    new_polygons = np.array([get_polygon(polygon) for polygon in polygons])
    new_polygons = new_polygons.reshape(-1, 8)
    labels.iloc[:, 1:] = new_polygons
    return labels



def get_polygon(points):
    """
    Rearranges four bounding box points into a counter-clockwise polygon order.

    Args:
        points (numpy.ndarray): A NumPy array of shape (4, 2) representing four points.

    Returns:
        numpy.ndarray: A NumPy array of shape (4, 2) containing the points in counter-clockwise order.

    Example:
        >>> points = np.array([[0.1, 0.2], [0.3, 0.1], [0.5, 0.4], [0.2, 0.5]])
        >>> ordered_points = get_polygon(points)
        >>> ordered_points
        array([[0.1, 0.2],
               [0.2, 0.5],
               [0.5, 0.4],
               [0.3, 0.1]])
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    top_left = points[np.argmin(x_coords + y_coords)]       # Minimize x + y
    top_right = points[np.argmin(-x_coords + y_coords)]     # Minimize -x + y
    bottom_left = points[np.argmin(x_coords - y_coords)]    # Minimize x - y
    bottom_right = points[np.argmin(-x_coords - y_coords)]  # Minimize -x - y
    return np.array([top_left, bottom_left, bottom_right, top_right])  # Counter-clockwise order




def augment(img, labels, img_path, label_path, args):
    """
    Augments an image and its labels based on specified augmentation parameters.

    This function generates augmented versions of an input image and its corresponding
    labels. It applies a series of augmentations (flip, rotate, grayscale, hue,
    saturation, and brightness) based on combinations generated by the
    'generate_augment_combos' function. It also saves the original and augmented images 
    and labels to a specified directory.
    If 'args.show' is True, it visualizes the augmentations.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
        img_path (str): The original image file path (used for saving augmented images).
        label_path (str): The original label file path (used for saving augmented labels).
        args (argparse.Namespace or similar): An object containing augmentation parameters.

    Returns:
        None: This function performs operations in place and saves files; it does not return a value.

    Note:
        This function assumes the existence of helper functions:
        - generate_augment_combos(args): Generates augmentation combinations.
        - save(img, labels, aug_num, img_path, label_path, args): Saves image and labels.
        - flip(img, labels, f): Applies flipping.
        - rotate(img, labels, r, args): Applies rotation.
        - grayscale(img, g): Applies grayscale conversion.
        - hue(img, h): Applies hue adjustment.
        - saturate(img, s): Applies saturation adjustment.
        - change_brightness(img, b): Applies brightness adjustment.
        - requantisize(labels): Applies requantization to labels.
        - rearrange(labels): Rearranges labels.
        - show_annotation(images_list, labels_list, agmnts_list, args): Visualizes augmentations.
    """
    combos = generate_augment_combos(args)
    aug_num = 0
    save(img, labels, aug_num, img_path, label_path, args) # save original image first in the new directory  
    images_list = [img]
    labels_list = [labels]
    agmnts_list = [None]
    for (f,r,g,h,s,b) in combos:
        new_img, new_labels = flip(img, labels, f)
        new_img, new_labels = rotate(new_img, new_labels, r, args)
        new_img = grayscale(new_img, g)
        new_img = hue(new_img, h)
        new_img = saturate(new_img, s)
        new_img = change_brightness(new_img, b)
        new_labels = requantisize(new_labels)
        new_labels = rearrange(new_labels)
        aug_num += 1
        save(new_img, new_labels, aug_num, img_path, label_path, args)
        images_list.append(new_img)
        labels_list.append(new_labels)
        agmnts_list.append((f,r,g,h,s,b))
    if args.show:
        show_annotation(images_list, labels_list, agmnts_list, args)
        



def generate_augment_combos(args):
    """
    Generates combinations of augmentation parameters based on provided arguments.
    
    Args:
        args (argparse.Namespace): An object containing augmentation parameters, including:
                                - flip (list): List of flip options.
                                - rotation (tuple): Rotation angle range (min, max).
                                - grayscale_chance (float): Probability of grayscale conversion.
                                - hue (tuple): Hue adjustment range (min, max).
                                - saturation (tuple): Saturation adjustment range (min, max).
                                - brightness (tuple): Brightness adjustment range (min, max).
                                - outputs_per_img (int): Number of augmentation combinations to generate.

    Returns:
        zip: A zip object containing tuples of augmentation parameters, where each tuple
             represents a combination of (flip, rotation, grayscale, hue, saturation, brightness).

    Example:
        >>> import argparse
        >>> args = argparse.Namespace(flip=['horizontal', 'vertical', None],
        ...                             rotation=(-10, 10),
        ...                             grayscale_chance=0.2,
        ...                             hue=(-20, 20),
        ...                             saturation=(-30, 30),
        ...                             brightness=(-15, 15),
        ...                             outputs_per_img=3)
        >>> combos = generate_augment_combos(args)
        >>> for combo in combos:
        ...     print(combo)
        ('horizontal', 1.234, 0, -5.678, 10.123, -2.345) # Example output; values will vary.
    """
    flp = random.choices(args.flip, k=args.outputs_per_img)
    rot = [random.uniform(args.rotation[0], args.rotation[1]) for _ in range(args.outputs_per_img)]
    gry = random.choices([0, 1], k=args.outputs_per_img, weights=[1 - args.grayscale_chance, args.grayscale_chance])
    hue = [random.uniform(args.hue[0], args.hue[1]) for _ in range(args.outputs_per_img)]
    sat = [random.uniform(args.saturation[0], args.saturation[1]) for _ in range(args.outputs_per_img)]
    brt = [random.uniform(args.brightness[0], args.brightness[1]) for _ in range(args.outputs_per_img)]
    return zip(flp, rot, gry, hue, sat, brt)
    


def flip(img, labels, option):
    """
    Flips an image and its corresponding labels based on the specified option.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
        option (str or None): The flip option. Can be 'vertical', 'horizontal', or None.

    Returns:
        tuple: A tuple containing the flipped image (NumPy array) and the adjusted labels
               (Pandas DataFrame). Returns copies of the original image and labels if option is None.
    """
    if option==None:
        return (img.copy(), labels.copy())
    if option=="vertical": # y-axis
        return flip_vertical(img, labels)
    if option=="horizontal": # x-axis
        return flip_horizontal(img, labels)
    


def flip_vertical(img, labels):
    """
    Flips an image vertically and adjusts the y-coordinates of its labels.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
                                     Expected columns: 'y1', 'y2', 'y3', 'y4'.

    Returns:
        tuple: A tuple containing the vertically flipped image (NumPy array) and the
               adjusted labels (Pandas DataFrame).
    """
    flipped_image = cv2.flip(img.copy(), 0)
    flipped_labels = labels.copy()
    flipped_labels[['y1', 'y2', 'y3', 'y4']] = 1.0 - flipped_labels[['y1', 'y2', 'y3', 'y4']]
    return flipped_image, flipped_labels



def flip_horizontal(img, labels):
    """
    Flips an image horizontally and adjusts the x-coordinates of its labels.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
                                     Expected columns: 'x1', 'x2', 'x3', 'x4'.

    Returns:
        tuple: A tuple containing the horizontally flipped image (NumPy array) and the
               adjusted labels (Pandas DataFrame).
    """
    flipped_image = cv2.flip(img.copy(), 1)
    flipped_labels = labels.copy()
    flipped_labels[['x1', 'x2', 'x3', 'x4']] = 1.0 - flipped_labels[['x1', 'x2', 'x3', 'x4']]
    return flipped_image, flipped_labels




def rotate(img, labels, angle, args):
    """
    Rotates an image and its corresponding labels by a specified angle.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        labels (pandas.DataFrame): The input labels as a Pandas DataFrame.
                                   Expected columns: 'label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'.
        angle (float): The rotation angle in degrees.
        args (argparse.Namespace): An object containing image size information
                                   ('img_size' as tuple: (height, width)).

    Returns:
        tuple: A tuple containing the rotated image (NumPy array) and the adjusted labels
               (Pandas DataFrame).
    """
    height, width = args.img_size
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)
    new_img = cv2.warpAffine(img.copy(), rotation_matrix, (width, height))
    new_labels = pd.DataFrame(columns=['label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
    for (c,x1,y1,x2,y2,x3,y3,x4,y4) in zip(labels['label'],labels['x1'],labels['y1'],labels['x2'],labels['y2'],labels['x3'],labels['y3'],labels['x4'],labels['y4']):
        x1, y1 = (x1*width, y1*height) 
        x2, y2 = (x2*width, y2*height) 
        x3, y3 = (x3*width, y3*height) 
        x4, y4 = (x4*width, y4*height) 
        points = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], dtype=np.float32)
        rotated_points = cv2.transform(points.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
        rotated_points = apply_bounds(rotated_points, args)
        [x1,y1,x2,y2,x3,y3,x4,y4] = rotated_points
        x1, y1 = (x1/width, y1/height) 
        x2, y2 = (x2/width, y2/height) 
        x3, y3 = (x3/width, y3/height) 
        x4, y4 = (x4/width, y4/height) 
        new_labels.loc[len(new_labels)] = [c,x1,y1,x2,y2,x3,y3,x4,y4]
    return (new_img, new_labels)



def apply_bounds(points, args):
    """
    Applies image boundary constraints to a set of points.

    Args:
        points (numpy.ndarray): A NumPy array of shape (N, 2) representing N points,
                                 where each point is [x, y].
        args (argparse.Namespace or similar): An object containing image size information
                                               ('img_size' as tuple: (height, width)).

    Returns:
        numpy.ndarray: A NumPy array containing the points with coordinates clipped
                       to the image boundaries.
    """
    new_points = np.array([])
    height, width = args.img_size
    for [x,y] in points:
        x = max(min(width, x), 0)
        y = max(min(height, y), 0)
        new_points = np.append(new_points, [x,y])
    return new_points



def grayscale(img, bit):
    """
    Converts an image to grayscale if the 'bit' flag is True.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        bit (bool): A boolean flag indicating whether to convert the image to grayscale.

    Returns:
        numpy.ndarray: The grayscale-converted image (BGR format) if 'bit' is True,
                       or the original image if 'bit' is False.
    """
    if bit:
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        new_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return new_img
    else:
        return img



def hue(img, hue_shift):
    """
    Adjusts the hue of an image by a specified shift value.

    Args:
        img (numpy.ndarray): The input image as a NumPy array (BGR format).
        hue_shift (float): The hue shift value (in degrees).

    Returns:
        numpy.ndarray: The hue-adjusted image as a NumPy array (BGR format).
    """
    hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def saturate(img, percent):
    """
    Adjusts the saturation of an image by a specified percentage.

    Args:
        img (numpy.ndarray): The input image as a NumPy array (BGR format).
        percent (float): The percentage by which to adjust the saturation.
                         Positive values increase saturation, negative values decrease.

    Returns:
        numpy.ndarray: The saturation-adjusted image as a NumPy array (BGR format).
    """
    saturation_factor = 1 + (percent/100)
    hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)



def change_brightness(img, percent):
    """
    Adjusts the brightness of an image by a specified percentage.

    Args:
        img (numpy.ndarray): The input image as a NumPy array (uint8).
        percent (float): The percentage by which to adjust the brightness.
                         Positive values increase brightness, negative values decrease.

    Returns:
        numpy.ndarray: The brightness-adjusted image as a NumPy array (uint8).
    """
    brightness_factor = 1 + (percent/100.0)
    new_img = np.clip(img.copy()*brightness_factor, 0, 255).astype(np.uint8)    
    return new_img



def requantisize(labels):
    """
    Requantizes the data types of label columns in a Pandas DataFrame.
    Args:
        labels (pandas.DataFrame): A DataFrame containing label data.
                                     Expected columns: 'label' and numeric columns.

    Returns:
        pandas.DataFrame: A DataFrame with the 'label' column as uint8 and the
                          numeric columns as float32.
    """
    new_labels = labels.copy()
    new_labels['label'] = new_labels['label'].astype(np.uint8)
    new_labels[new_labels.columns[1:]] = new_labels[new_labels.columns[1:]].astype(np.float32)
    return new_labels




def show_annotation(images_list, labels_list, augments_list, args):
    """
    Displays images with bounding box annotations and augmentation information. The points for 
    each annotation follows a counter-clockwise direction.

    Args:
        images_list (list): A list of NumPy arrays representing images.
        labels_list (list): A list of Pandas DataFrames, where each DataFrame contains
                            bounding box labels for the corresponding image.
        augments_list (list): A list of tuples, where each tuple contains augmentation
                              parameters (flip, rotation, grayscale, hue, saturation, brightness)
                              for the corresponding image. None if no augmentation.
        args (argparse.Namespace): An object containing image size information
                                   ('img_size' as tuple: (height, width)).

    Returns:
        None: This function displays images using OpenCV; it does not return a value.
    """
    for orig_img, orig_labels, augment in zip(images_list, labels_list, augments_list):
        img = orig_img.copy()
        labels = orig_labels.copy()
        height, width = args.img_size
        for _, coord in labels.iterrows():
            p1 = [round(coord['x1'] * width), round(coord['y1'] * height)] 
            p2 = [round(coord['x2'] * width), round(coord['y2'] * height)] 
            p3 = [round(coord['x3'] * width), round(coord['y3'] * height)] 
            p4 = [round(coord['x4'] * width), round(coord['y4'] * height)] 
            points = np.array([p1, p2, p3, p4]).astype(np.int32)
            texts = ['P1', 'P2', 'P3', 'P4']
            for text, point in zip(texts, points):
                x = max(min(width-5, point[0]), 20)
                y = max(min(height-5, point[1]), 20)
                cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            reshaped_points = points.reshape((-1, 1, 2))
            cv2.polylines(img, [reshaped_points], isClosed=True, color=(0,255,0), thickness=2)
            if augment is not None:
                (f,r,g,h,s,b) = augment
                cv2.putText(img, f"Flip: {f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, "Rotation: {:.2f} degrees".format(r), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, "Grayscale: {}".format(True if g else False), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, "Hue: {:.2f} degrees".format(h), (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, "Saturation: {:.2f}%".format(s), (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(img, "Brightness: {:.2f}%".format(b), (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        img = cv2.resize(img, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Ground Truth", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()




def save(new_img, new_labels, img_num, img_path, lbl_path, args):
    """
    Saves an image and its corresponding labels to specified output directories.
    The image is saved as a PNG file, and the labels are saved as a space-separated TXT file.

    Args:
        new_img (numpy.ndarray): The image to save as a NumPy array.
        new_labels (pandas.DataFrame): The labels to save as a Pandas DataFrame.
        img_num (int): An integer representing the image number (used for file naming).
        img_path (Path or str): The original image file path.
        lbl_path (Path or str): The original label file path.
        args (argparse.Namespace or similar): An object containing the destination path
                                               ('destination_path' as Path or str).

    Returns:
        None: This function saves files to disk; it does not return a value.
    """
    out_img_path = Path(args.destination_path) / img_path.parent.parent.name / 'images'
    out_lbl_path = Path(args.destination_path) / lbl_path.parent.parent.name / 'labels'
    check_directories(out_img_path)
    check_directories(out_lbl_path)
    img_fname = img_path.name.split('.')[0] + f'_{img_num:0{4}d}.PNG'
    lbl_fname = lbl_path.name.split('.')[0] + f'_{img_num:0{4}d}.TXT'
    cv2.imwrite(out_img_path / img_fname, new_img)
    new_labels.to_csv(out_lbl_path / lbl_fname, sep=" ", header=False, index=False, float_format="%.6f")
    if args.show: 
        print(f"Saving {out_img_path / img_fname}")
    



def check_directories(directory_path: Path):
    """
    Checks if a directory exists and creates it if it doesn't.

    Args:
        directory_path (Path): A Path object representing the directory path.

    Returns:
        None: This function modifies the filesystem; it does not return a value.
    """
    directory_path = directory_path.resolve()  # Ensure absolute path
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")


def get_images_paths_on_one_folder(set_path):
    """
    Retrieves the file paths of all images within a specified folder.

    Args:
        set_path (str or Path): The path to the parent folder containing the 'images' subfolder.

    Returns:
        list[Path]: A list of Path objects, where each Path object represents the
                    file path of an image within the 'images' subfolder.
 """
    path = Path(set_path) / 'images'
    images_fnames = [f.name for f in path.iterdir() if f.is_file()]
    images_paths = [path / fname for fname in images_fnames]
    return images_paths



def get_images_paths_on_set_folders(parent_path):
    """
    Retrieves the file paths of all images within 'train', 'val', and 'test' subfolders.

    Args:
        parent_path (str or Path): The path to the parent folder containing the
                                     'train', 'val', and 'test' subfolders.

    Returns:
        list[Path]: A list of Path objects, where each Path object represents the
                    file path of an image within the 'images' subfolders of
                    'train', 'val', or 'test'.
    """
    sets = ['train', 'val', 'test']
    images_paths = []
    for set_folder in sets:
        images_path = Path(parent_path) / set_folder / 'images'
        images_fnames = [f.name for f in images_path.iterdir() if f.is_file()]
        images_paths_set = [images_path / fname for fname in images_fnames]
        images_paths+= images_paths_set
    return images_paths



def get_labels_paths(imgs_paths):
    """
    Retrieves the corresponding label file paths for a list of image file paths.

    Args:
        imgs_paths (list[Path]): A list of Path objects representing image file paths.

    Returns:
        list[Path]: A list of Path objects representing the corresponding label file paths.
    """
    labels_paths = []
    for img_path in imgs_paths:
        label_path = img_path.parent.parent / 'labels' / img_path.name 
        label_path = label_path.with_suffix(".txt")
        labels_paths.append(label_path)
    return labels_paths



def load_yaml(filepath):
    """Loads a YAML file and returns its content as a Python dictionary."""
    try:
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)  # Use safe_load to prevent security risks
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This program performs data augmentation for segmentation task where each annotation is a 4-point polygon.")
    parser.add_argument('--data', required=True, help="Path to the configuration file containing the details such as paths and augmentation parameters.")
    parser.add_argument('--show', action='store_true', help="Whether to visualize augmentation.")
    parser.add_argument('--no-show', action='store_false', dest='show', help="Don't visualize augmentation.")
    parser.set_defaults(show=True)
    parser_args = parser.parse_args()



    yaml_data = load_yaml(parser_args.data) 
    if yaml_data:
        args = argparse.Namespace(**yaml_data, show=parser_args.show)
    

    images_paths = get_images_paths_on_set_folders(args.source_path)
    labels_paths = get_labels_paths(images_paths)

    
    if not args.show:
        progress_bar = tqdm(total=len(images_paths))
    for img_path, label_path in zip(images_paths, labels_paths):
        img, labels = load(img_path, label_path)
        img, labels = preprocess(img, labels, args)
        augment(img, labels, img_path, label_path, args)
        if not args.show:
            progress_bar.update(1)