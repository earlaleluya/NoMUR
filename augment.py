import cv2
import os
import pandas as pd
import random
import numpy as np
import argparse
from pathlib import Path 



def load(img_path, label_path):
    bgr_img = cv2.imread(img_path)
    labels = load_labels(label_path)
    return (bgr_img, labels)


def load_labels(labels_txt_file_path):
    column_names = ['label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']    
    dtype_mapping = {col: np.float32 for col in column_names[1:]}  # All except 'label' as float
    dtype_mapping['label'] = np.int32  # 'label' as int
    df = pd.read_csv(labels_txt_file_path, sep=r'\s+', names=column_names, dtype=dtype_mapping)
    return df


def preprocess(img, labels, args):
    h, w = args.img_size
    new_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    new_labels = labels[labels['label'].isin(args.classes)]
    new_labels = rearrange(new_labels)
    return (new_img, new_labels)



def rearrange(labels):
    polygons = np.array(labels[['x1','y1','x2','y2','x3','y3','x4','y4']])
    polygons = polygons.reshape(-1, 4, 2)
    new_polygons = np.array([get_polygon(polygon) for polygon in polygons])
    new_polygons = new_polygons.reshape(-1, 8)
    labels.iloc[:, 1:] = new_polygons
    return labels



def get_polygon(points):
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    top_left = points[np.argmin(x_coords + y_coords)]       # Minimize x + y
    top_right = points[np.argmin(-x_coords + y_coords)]     # Minimize -x + y
    bottom_left = points[np.argmin(x_coords - y_coords)]    # Minimize x - y
    bottom_right = points[np.argmin(-x_coords - y_coords)]  # Minimize -x - y
    return np.array([top_left, bottom_left, bottom_right, top_right])  # Counter-clockwise order




def augment(img, labels, img_path, label_path, args):
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
    flp = random.choices(args.flip, k=args.outputs_per_img)
    rot = [random.uniform(args.rotation[0], args.rotation[1]) for _ in range(args.outputs_per_img)]
    gry = random.choices([0, 1], k=args.outputs_per_img, weights=[1 - args.grayscale_chance, args.grayscale_chance])
    hue = [random.uniform(args.hue[0], args.hue[1]) for _ in range(args.outputs_per_img)]
    sat = [random.uniform(args.saturation[0], args.saturation[1]) for _ in range(args.outputs_per_img)]
    brt = [random.uniform(args.brightness[0], args.brightness[1]) for _ in range(args.outputs_per_img)]
    return zip(flp, rot, gry, hue, sat, brt)
    

def flip(img, labels, option):
    if option==None:
        return (img.copy(), labels.copy())
    if option=="vertical": # y-axis
        return flip_vertical(img, labels)
    if option=="horizontal": # x-axis
        return flip_horizontal(img, labels)
    

def flip_vertical(img, labels):
    flipped_image = cv2.flip(img.copy(), 0)
    flipped_labels = labels.copy()
    flipped_labels[['y1', 'y2', 'y3', 'y4']] = 1.0 - flipped_labels[['y1', 'y2', 'y3', 'y4']]
    return flipped_image, flipped_labels



def flip_horizontal(img, labels):
    flipped_image = cv2.flip(img.copy(), 1)
    flipped_labels = labels.copy()
    flipped_labels[['x1', 'x2', 'x3', 'x4']] = 1.0 - flipped_labels[['x1', 'x2', 'x3', 'x4']]
    return flipped_image, flipped_labels


def rotate(img, labels, angle, args):
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
    new_points = np.array([])
    height, width = args.img_size
    for [x,y] in points:
        x = max(min(width, x), 0)
        y = max(min(height, y), 0)
        new_points = np.append(new_points, [x,y])
    return new_points


def grayscale(img, bit):
    if bit:
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        new_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return new_img
    else:
        return img


def hue(img, hue_shift):
    hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def saturate(img, percent):
    saturation_factor = 1 + (percent/100)
    hsv_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def change_brightness(img, percent):
    brightness_factor = 1 + (percent/100.0)
    new_img = np.clip(img.copy()*brightness_factor, 0, 255).astype(np.uint8)    
    return new_img


def show_annotation(images_list, labels_list, augments_list, args):
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
    out_img_path = Path(args.out_path) / img_path.parent.parent.name / 'images'
    out_lbl_path = Path(args.out_path) / lbl_path.parent.parent.name / 'labels'
    check_directories(out_img_path)
    check_directories(out_lbl_path)
    img_fname = img_path.name.split('.')[0] + f'_{img_num:0{4}d}.PNG'
    lbl_fname = lbl_path.name.split('.')[0] + f'_{img_num:0{4}d}.TXT'
    cv2.imwrite(out_img_path / img_fname, new_img)
    new_labels.to_csv(out_lbl_path / lbl_fname, sep=" ", header=False, index=False, float_format="%.6f")
    print(f"Saving {out_img_path / img_fname}")
    


def check_directories(directory_path: Path):
    directory_path = directory_path.resolve()  # Ensure absolute path
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{directory_path}' created.")



def requantisize(labels):
    new_labels = labels.copy()
    new_labels['label'] = new_labels['label'].astype(np.uint8)
    new_labels[new_labels.columns[1:]] = new_labels[new_labels.columns[1:]].astype(np.float32)
    return new_labels


def get_images_paths_on_one_folder(set_path):
    path = Path(set_path) / 'images'
    images_fnames = [f.name for f in path.iterdir() if f.is_file()]
    images_paths = [path / fname for fname in images_fnames]
    return images_paths


def get_images_paths_on_set_folderss(parent_path):
    sets = ['train', 'val', 'test']
    images_paths = []
    for set_folder in sets:
        images_path = Path(parent_path) / set_folder / 'images'
        images_fnames = [f.name for f in images_path.iterdir() if f.is_file()]
        images_paths_set = [images_path / fname for fname in images_fnames]
        images_paths+= images_paths_set
    return images_paths


def get_labels_paths(imgs_paths):
    labels_paths = []
    for img_path in imgs_paths:
        label_path = img_path.parent.parent / 'labels' / img_path.name 
        label_path = label_path.with_suffix(".txt")
        labels_paths.append(label_path)
    return labels_paths




if __name__ == "__main__":

    args = argparse.Namespace(
        inp_path = "D:/Virtual Space/Desktop/mel_ral/Aerial Based Pavement Distress Detection Method for Unstructured Road Codes and Datasets/Dataset/NoMUR Datasets/NoMURLane_orig",
        out_path = "D:/Virtual Space/Desktop/mel_ral/Aerial Based Pavement Distress Detection Method for Unstructured Road Codes and Datasets/Dataset/NoMUR Datasets/NoMURLane",
        img_size = (1088, 1920), # (height, width)
        id2class = {0 : "road_lane"},
        classes = [0],          # state which class to augment
        outputs_per_img = 20,   # number of augmented version out of one raw image
        flip = ['horizontal', 'vertical', None],
        rotation = [-10, 10],   # angle ranges
        grayscale_chance = 0.4, # chance of generating grayscale augmentations
        hue = [-50, 50],        # percentage ranges
        saturation = [-50, 50], # percentage ranges
        brightness = [-50, 50], # percentage ranges
        show=True,              # bool if to visualize images
    )


    images_paths = get_images_paths_on_set_folderss(args.inp_path)
    labels_paths = get_labels_paths(images_paths)

    for img_path, label_path in zip(images_paths, labels_paths):
        img, labels = load(img_path, label_path)
        img, labels = preprocess(img, labels, args)
        augment(img, labels, img_path, label_path, args)
