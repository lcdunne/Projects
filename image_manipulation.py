'''
Code to perform:
1. Convert images to greyscale
2. Convert images to edges (and optional invert colour) - line drawings.
3. Resizing of images
'''
import numpy as np
import cv2
import os


def convert_to_greyscale(fpath, save=False, suffix='bw'):
    # Read it
    img = cv2.imread(fpath)
    greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save == True:
        path = '/'.join(fpath.split('/')[:-1])
        fname = fpath.split('/')[-1]
        f, e = fname.split('.')  # Split name from extension
        f = f + '_' + suffix  # Append suffix to the name
        fname = '.'.join([f, e])  # Join it back up
        output_path = f'{path}/{fname}'  # Join it to the path for outputting
        cv2.imwrite(output_path, greyed)
    return greyed


def convert_to_edges(fpath, invert=False, thresh=(100, 200), save=False, suffix='edges'):
    # Read it
    img = cv2.imread(fpath)
    edges = cv2.Canny(img, thresh[0], thresh[1])  # specify minimum and maximum threshold
    # The lower value is a strict cutoff, below which all values are rejected regardless of continuity
    # The upper value is less strict. If pixels fall between, they are generally rejected unless...
    #   they follow a continuation up and beyond the max value in which case they are accepted.
    # Invert?
    if invert == True:
        edges = cv2.bitwise_not(edges)

    if save == True:
        path = '/'.join(fpath.split('/')[:-1])
        fname = fpath.split('/')[-1]
        f, e = fname.split('.')  # Split name from extension
        f = f + '_' + suffix  # Append suffix to the name
        fname = '.'.join([f, e])  # Join it back up
        output_path = f'{path}/{fname}'  # Join it to the path for outputting
        cv2.imwrite(output_path, edges)
    return edges


def resize_image(fpath, dims, save=False, suffix='resized'):
    # Read it
    img = cv2.imread(fpath)
    resized = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)

    if save == True:
        path = '/'.join(fpath.split('/')[:-1])
        fname = fpath.split('/')[-1]
        f, e = fname.split('.')  # Split name from extension
        f = f + '_' + suffix  # Append suffix to the name
        fname = '.'.join([f, e])  # Join it back up
        output_path = f'{path}/{fname}'  # Join it to the path for outputting
        cv2.imwrite(output_path, resized)
    return resized


if __name__ == '__main__':
    # Set the image path
    path = 'Path/To/Your/Images'
    allfiles = os.listdir(path)
    for i, f in enumerate(allfiles):
        print(f"Converting {f.split('.')[0]}\t{i+1} of {len(allfiles)} ({((i+1) / len(allfiles)) * 100}%")
        fpath = f'{path}/{f}'
        convert_to_greyscale(fpath=fpath, suffix='', save=True)
        # convert_to_edges(fpath=fpath, invert=True, thresh=(110, 210), save=True)
        # resize_image(fpath=fpath, dims=(200, 200), save=True, suffix='')
