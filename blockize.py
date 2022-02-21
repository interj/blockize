#!/usr/bin/python3
import cv2, argparse, math
import numpy as np
from os import path
from collections import deque, defaultdict
import random
MAX_ALPHA = 255

parser = argparse.ArgumentParser(description='Convert an image to JSON serialized 3d object made out of blocks')
parser.add_argument('filepath', metavar='image_path', type=path.abspath, help='image file to be converted')
parser.add_argument('--size', '-s', metavar='NUM', default=2, type=float, 
                    help='final asset size along the longer side, defaults to 2.0')
parser.add_argument('--colors', '-c', metavar='NUM', default=24, type=int, help= 'amount of colors to use, defaults to 24')
parser.add_argument('--pixels', '-p', metavar='NUM', default=40000, type=int, 
                    help='maximum count of input pixels to k-means, defaults to 40000')

args = parser.parse_args()


def kmeans_color_quantization(image, clusters=32, rounds=8):
    h, w, depth = image.shape
    samples = np.zeros([h*w, depth], dtype=np.float32)
    count = 0

    for y in range(h):
        for x in range(w):
            samples[count] = image[y,x]
            if depth == 4:
                a = samples[count][3]
                if a > MAX_ALPHA * 2/3:
                    samples[count][3] = MAX_ALPHA
                elif a < MAX_ALPHA * 1/3:
                    samples[count][3] = 0
                else:
                    samples[count][3] = MAX_ALPHA * 0.5
            count += 1
    
    print('Copy of', w*h, 'pixels done')

    compactness, labels, centers = cv2.kmeans(
        samples,
        clusters, 
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), 
        rounds, 
        cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))
    
def represent_with_squares(image):
    h, w, depth = image.shape
    pixels = defaultdict(deque)
    
    x = y = 0
    while x < w or y < h:
        if x < w:
            for scanned_y in range(y):
                pixels[image[scanned_y, x].tobytes()].append((scanned_y, x))
            x += 1
        if y < h:
            for scanned_x in range(x):
                pixels[image[y, scanned_x].tobytes()].append((y, scanned_x))
            y += 1

    pixels = dict(sorted(pixels.items(), key=lambda item: len(item[1]), reverse = True))
    l = list(pixels.items())
    top = l[:8]
    random.shuffle(top)
    l[:8] = top
    pixels = dict(l)    
    layers = dict()
    i = 0
    for color in pixels:
        layers[color] = i
        i += 1
    
    rects = defaultdict(deque)
    #most populous color is a rectangle the size of canvas, optimization disabled if alpha
    start_from = 0
    if depth < 4:
        rects[next(iter(pixels))].append(((0,0), (h, w)))
        start_from = 1
    
    for color in list(pixels.keys())[start_from:]:
        x_can_grow = y_can_grow = True
        y, x = pixels[color][0]
        start_y = y
        start_x = x
        #print(pixels[color])
        while len(pixels[color]) > 0:
            if not ((x_can_grow and x < w) or (y_can_grow and y < h)):
                x_can_grow = y_can_grow = True
                y, x = pixels[color][0]
                start_y = y
                start_x = x
            #print('x:', x, x_can_grow, 'y:', y, y_can_grow)
            possible_pops = list()
            if x < w and x_can_grow:
                for scanned_y in range(start_y, y):
                    if image[scanned_y, x].tobytes() == color:
                        possible_pops.append((scanned_y, x))
                    elif layers[image[scanned_y, x].tobytes()] < layers[color]: # <
                        x_can_grow = False
                        possible_pops = list()
                        x -= 1
                        break;
                #print(possible_pops)
                for pop_y, pop_x in possible_pops:
                    try:
                        pixels[color].remove((pop_y, pop_x))
                    except:
                        pass
                x += 1
                possible_pops = list()

            if y < h and y_can_grow:
                for scanned_x in range(start_x, x):
                    if image[y, scanned_x].tobytes() == color:
                        possible_pops.append((y, scanned_x))
                    elif layers[image[y, scanned_x].tobytes()] < layers[color]: # <
                        y_can_grow = False
                        possible_pops = list()
                        y -= 1
                        break;
                #print(possible_pops)
                for pop_y, pop_x in possible_pops:
                    try:
                        pixels[color].remove((pop_y, pop_x))
                    except:
                        pass
                y += 1
                possible_pops = list()
            if not ((x_can_grow and x < w) or (y_can_grow and y < h)):
                rects[color].append(((start_y, start_x), (y, x)))
        
        rects[color].append(((start_y, start_x), (y, x)))
    return rects


img = cv2.imread(args.filepath, cv2.IMREAD_UNCHANGED)
print('Original Dimensions: ', img.shape)

MAX_INPUT_PIXELS = args.pixels
scale_factor = max(math.sqrt(img.shape[0] * img.shape[1] / MAX_INPUT_PIXELS), 1)
width = int(img.shape[1] / scale_factor)
height = int(img.shape[0] / scale_factor)
dim = (width, height)

  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
print('Resized Dimensions: ', img.shape)
cv2.imwrite('resized' + path.basename(args.filepath), img)


img = kmeans_color_quantization(img, clusters = args.colors)
print('K-means done with', args.colors, 'clusters')
cv2.imwrite('flattened2' + path.basename(args.filepath), img)

rects = represent_with_squares(img)
print('Rect count:', sum(len(rects[key]) for key in rects))
 
start_x=-4
start_y= 8
start_z=-2.0001
group_start='{"valuetype":"float","objects":[{"n":"Box","p":[-6,12,-3],"r":[90,0,0],"s":[10,1,10],"c":[1,0,0]},{"n":"group","objects":['
box='{"n":"Box","p":[%f,%f,%f],"r":[90,0,0],"s":[%f,%f,%f],"c":[%f,%f,%f]},'
semi_transparent='{"n":"Box","p":[%f,%f,%f],"r":[90,0,0],"s":[%f,%f,%f],"c":[%f,%f,%f],"m":"glass_2"},'
group_end=']}]}'

pixel_size = args.size / max(height, width)

with open(path.basename(args.filepath) + '.json', 'w') as assetFile:
    assetFile.write(group_start)
    i = 1
    for key in rects:
        b, g, r, *a = np.frombuffer(key, dtype = np.uint8)
        depth = 0.001 / len(rects) * i + 0.01
        i += 1
        for rect in rects[key]:
            (y, x), (end_y, end_x) = rect
            rect_w = (end_x - x) * pixel_size
            rect_h = (end_y - y) * pixel_size
            
            json = box
            if len(a) == 1 and a[0] == 0:
                continue
            elif len(a) == 1 and a[0] == MAX_ALPHA * 0.5:
                json = semi_transparent
            
            assetFile.write(json % (start_x - x * pixel_size - rect_w / 2, start_y - y * pixel_size, start_z + depth/2, rect_w, depth, rect_h, r/255, g/255, b/255))
    
    assetFile.write(group_end)
