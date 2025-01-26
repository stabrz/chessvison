# %%


# %%

import cv2
import os

def get_images(image_dir, label_dir):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        label_filename = '\\' + filename[:-4] + '.txt'
        label_path = os.path.join(label_dir + label_filename)
        image = (cv2.imread(image_path))
        images.append(image)
        with open(label_path, 'r') as f:
            label_data = f.readlines() 
            labelz = []
            for line in label_data:
                label = line.strip().split(" ")
                label = int(label[0])
                labelz.append(label)
            labels.append(labels)
    return images, labels

train_image_dir = r"train\images"
test_image_dir = r"test\images"
valid_image_dir = r"valid\images"
train_label_dir = r"train\labels"
test_label_dir = r"test\labels"
valid_label_dir = r"valid\labels"

train_im, train_lab = get_images(train_image_dir, train_label_dir)
test_im, test_lab = get_images(test_image_dir, test_label_dir)
val_im, val_lab = get_images(valid_image_dir, valid_label_dir)

sample_image = cv2.imread(r"train\images\c46bf04050a2a9323dfe563e8813602f_jpg.rf.3c67770c7bc1ae2f34811acc4fea44c1.jpg")


# %%

import numpy as np
def create_green_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  
    lower_green = np.array([40, 40, 40])  
    upper_green = np.array([90, 255, 255]) 

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    green_image = cv2.bitwise_and(image, image, mask=green_mask)

    return green_mask, green_image

mask, green_image = create_green_mask(sample_image)


# %%

import matplotlib.pyplot as plt
mask, green_image = create_green_mask(sample_image)

mask = mask.copy()

def Harris_corner_detector(image, threshold):
    processed_image = image.copy()
    if len(processed_image.shape) == 2 or processed_image.shape[2] == 1:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray, 2, 5, 0.04)
    dst = cv2.dilate(dst, None)
    
    corners = np.argwhere(dst > threshold * dst.max())
    
    corners = [tuple(c[::-1]) for c in corners] 
    
    return processed_image, corners

def filter_chessboard_corners(corners, min_distance):
    filtered_corners = []

    for corner in corners:
        if all(np.linalg.norm(np.array(corner) - np.array(fc)) > min_distance for fc in filtered_corners):
            filtered_corners.append(corner)

    return filtered_corners



image = mask  
threshold = 0.1 

result_image, corners = Harris_corner_detector(image, threshold)
best_image = result_image.copy()

min_distance = 10  
filtered_corners = filter_chessboard_corners(corners, min_distance)


for corner in filtered_corners:
    cv2.circle(result_image, corner, 3, (0, 255, 0), -1)  




# %%

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Assuming result_image and filtered_corners are defined
# Define how many points you want to use for line fitting
num_points = 5  # Desired number of points
distance_threshold = 30  # Minimum distance between points

def filter_points(points, threshold):
    filtered_points = []
    for point in points:
        if all(np.linalg.norm(np.array(point) - np.array(existing_point)) >= threshold for existing_point in filtered_points):
            filtered_points.append(point)
        if len(filtered_points) == num_points:
            break
    return filtered_points

# Sort corners by y-coordinate
sorted_by_y = sorted(filtered_corners, key=lambda corner: corner[1])  # Sort by y
lowest_y_points = filter_points(sorted_by_y, distance_threshold)  # Points with lowest y
highest_y_points = filter_points(sorted_by_y[::-1], distance_threshold)  # Points with highest y

# Sort corners by x-coordinate
sorted_by_x = sorted(filtered_corners, key=lambda corner: corner[0])  # Sort by x
lowest_x_points = filter_points(sorted_by_x, distance_threshold)  # Points with lowest x
highest_x_points = filter_points(sorted_by_x[::-1], distance_threshold)  # Points with highest x
# print(lowest_x_points)

# Print the results
print("Lowest Y points:", lowest_y_points)
print("Highest Y points:", highest_y_points)
print("Lowest X points:", lowest_x_points)
print("Highest X points:", highest_x_points)

# Function to fit a line, draw it and return the equation coefficients
# def draw_line(points, image, color):
#     if len(points) < 2:  # Need at least 2 points to fit a line
#         return None
#     x_coords, y_coords = zip(*points)
#     coefficients = np.polyfit(x_coords, y_coords, 1)  # Linear regression
#     slope, intercept = coefficients
#     print(slope, intercept)

#     height, width = image.shape[:2]
#     # Calculate intersection points with image borders
#     x0, y0 = 0, int(intercept)
#     x1, y1 = width, int(slope * width + intercept)

#     # Draw the line on the image
#     cv2.line(image, (x0, y0), (x1, y1), color, 2)

#     return (slope, intercept)  # Return line equation coefficients for further use
def draw_line(points, image, color, orientation):
    if len(points) < 2:  # Need at least 2 points to fit a line
        return None

    min_error = float('inf')
    best_coefficients = None
    best_slope = None

    # Try excluding two points at a time
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            subset = [p for k, p in enumerate(points) if k != i and k != j]
            x_coords, y_coords = zip(*subset)
            coefficients = np.polyfit(x_coords, y_coords, 1)  # Linear regression
            slope, intercept = coefficients

            # Compute error as the sum of squared residuals
            error = sum((y - (slope * x + intercept)) ** 2 for x, y in subset)

            # Check slope constraint based on orientation
            if orientation == 'horizontal' and abs(slope) >= 1:
                continue
            if orientation == 'vertical' and abs(slope) <= 4:
                continue

            if error < min_error:
                min_error = error
                best_coefficients = (slope, intercept)
                # print(subset)
                # print(best_coefficients)

    if best_coefficients is None:
        # Default to a parallel or perpendicular line through the average point
        x_coords, y_coords = zip(*points)
        avg_x = sum(x_coords) / len(x_coords)
        avg_y = sum(y_coords) / len(y_coords)

        if orientation == 'horizontal':
            slope = 0  # Horizontal line
            intercept = avg_y
        elif orientation == 'vertical':
            slope = 6  # Large slope for near-vertical line
            intercept = -slope * avg_x + avg_y
        best_coefficients = (slope, intercept)

    # slope, intercept = best_coefficients
    if slope == 0:
        best_coefficients = (1e-5, intercept)
    # print(best_coefficients)

    height, width = image.shape[:2]

    if abs(slope) > 1e5:  # Near-vertical line
        x0, y0 = int(-intercept / slope), 0
        x1, y1 = int(-intercept / slope), height
    else:  # Non-vertical line
        x0, y0 = 0, int(intercept)
        x1, y1 = width, int(slope * width + intercept)

#     # Draw the line on the image
#     height, width = image.shape[:2]
# #     # Calculate intersection points with image borders
#     x0, y0 = 0, int(intercept)
#     x1, y1 = width, int(slope * width + intercept)
    cv2.line(image, (x0, y0), (x1, y1), color, 2)

    return best_coefficients


# Collect the equations of the lines
line_equations = {}


# Mark the points and draw lines
line_equations['lowest_y'] = draw_line(lowest_y_points, result_image, (255, 0, 0), 'horizontal')
for point in lowest_y_points:
    cv2.circle(result_image, point, 5, (255, 0, 0), -1)  # Mark lowest Y points in blue

line_equations['highest_y'] = draw_line(highest_y_points, result_image, (0, 0, 255), 'horizontal')
for point in highest_y_points:
    cv2.circle(result_image, point, 5, (0, 0, 255), -1)  # Mark highest Y points in red

line_equations['lowest_x'] = draw_line(lowest_x_points, result_image, (0, 255, 0), 'vertical')
for point in lowest_x_points:
    cv2.circle(result_image, point, 5, (0, 255, 0), -1)  # Mark lowest X points in green

line_equations['highest_x'] = draw_line(highest_x_points, result_image, (0, 255, 255), 'vertical')
for point in highest_x_points:
    cv2.circle(result_image, point, 5, (0, 255, 255), -1)  # Mark highest X points in yellow

# Display the image with marked points and lines


print(line_equations)


# %%
x = [36, 46, 48]
y = [314, 205, 371]
licznik = 0
mianownik = 0
for i in range(3):
    licznik +=((x[i]-np.average(x))*(y[i]-np.average(y)))
    mianownik += (x[i]-np.average(x))**2
alfa = licznik/mianownik
beta = np.average(y) - alfa*np.average(x)
print(alfa, beta)


# %%
print (line_equations)

def find_intersection_points(line_equations):
    intersection_points = {}
    for (name1, eq1), (name2, eq2) in [
        (a, b) for a in line_equations.items() for b in line_equations.items() if a != b
    ]:
        try:
            # Handle potential division by zero
            if eq1[0] == eq2[0]:  # Parallel lines
                continue

            x = (eq2[1] - eq1[1]) / (eq1[0] - eq2[0])
            y = eq1[0] * x + eq1[1]

            # Handle potential NaN or infinite values
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            intersection_points[(name1, name2)] = (int(round(x)), int(round(y)))

        except Exception as e:
            # Handle any unexpected errors
            print(f"Error processing lines {name1} and {name2}: {e}")
            continue

    return intersection_points

# Find the intersection points
intersection_points = find_intersection_points(line_equations)
print(intersection_points)



# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Assuming intersection_points is already defined
# Define boundaries
min_boundary = -200
max_boundary = 800

intersection_points = intersection_points.values()
# Filter and remove duplicates
filtered_intersection_points = list(set(
    tuple(point) for point in intersection_points 
    if min_boundary <= point[0] <= max_boundary and min_boundary <= point[1] <= max_boundary
))

print(filtered_intersection_points)

# Mark the filtered intersections on the image
for point in filtered_intersection_points:
    cv2.circle(best_image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # Mark in white

# Display the image with lines and intersections

# assign lebels to the corners(topleft, topright, bottomleft, bottomright)(0,1,2,3)

# Sort the filtered intersection points by x and y coordinates

# def _lenth_from0_0(intersection_points):
#     sorted_intersection_points = []
#     for point in intersection_points:
#         distance = np.sqrt(point[0]**2 + point[1]**2)
#         sorted_intersection_points.append((point, distance))
#         # do not return distance
#         sorted_intersection_points.sort(key=lambda x: x[1])
#     return sorted_intersection_points

# sorted_intersection_points = _lenth_from0_0(filtered_intersection_points)

# print(sorted_intersection_points)

# # Assign labels to the corners
# topleft = sorted_intersection_points[0][0]
# topright = sorted_intersection_points[1][0]
# bottomleft = sorted_intersection_points[2][0]
# bottomright = sorted_intersection_points[3][0]

# print("Top Left:", topleft)
# print("Top Right:", topright)
# print("Bottom Left:", bottomleft)
# print("Bottom Right:", bottomright)

def assign_corners(points):
    """
    Assigns a set of 4 points to top-left, top-right, bottom-left, and bottom-right.

    Args:
        points (list of tuple): A list of 4 points, where each point is a tuple (x, y).

    Returns:
        dict: A dictionary with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'.
    """
    if len(points) < 4:
        raise ValueError("At least 4 points are required.")

    # If more than 4 points, reduce to 4 by removing closest pairs
    while len(points) > 4:
        min_distance = float('inf')
        pair_to_remove = None

        # Find the pair of points with the smallest distance
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = points[i], points[j]
                distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    pair_to_remove = j  # Index of the second point in the closest pair

        # Remove one of the points in the closest pair
        if pair_to_remove is not None:
            points.pop(pair_to_remove)

    # Sort points first by y-coordinate, then by x-coordinate
    points_sorted = sorted(points, key=lambda p: (p[1], p[0]))

    # Top two points (lowest y-coordinates)
    top_two = points_sorted[:2]
    # Bottom two points (highest y-coordinates)
    bottom_two = points_sorted[2:]

    # Sort top two by x to distinguish left and right
    top_left, top_right = sorted(top_two, key=lambda p: p[0])
    # Sort bottom two by x to distinguish left and right
    bottom_left, bottom_right = sorted(bottom_two, key=lambda p: p[0])

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }




points = assign_corners(filtered_intersection_points)
topleft = points['top_left']
topright = points['top_right']
bottomleft = points['bottom_left']
bottomright = points['bottom_right']

print("Top Left:", topleft)
print("Top Right:", topright)
print("Bottom Left:", bottomleft)
print("Bottom Right:", bottomright)

# without distance from 0,0




# %%

chessboard_boundaries = {
    'top-left': topleft,
    'top-right': topright,
    'bottom-left': bottomleft,
    'bottom-right': bottomright
}


def boundaries_to_vector(boundaries):
    order = ['top-left', 'top-right', 'bottom-right', 'bottom-left']

    vector = []

    for corner in order:
        vector.extend(boundaries[corner])

    return vector

chessboard_vector = boundaries_to_vector(chessboard_boundaries)



target_points = np.array([
    [0, 0],
    [416, 0],
    [416, 416],
    [0, 416]
])

chessboard_points = np.array([chessboard_boundaries[corner] for corner in ['top-left', 'top-right', 'bottom-right', 'bottom-left']])

transformation_matrix = cv2.getPerspectiveTransform(chessboard_points.astype(np.float32), target_points.astype(np.float32))


transformed_image = cv2.warpPerspective(best_image, transformation_matrix, (416, 416))




# %%

def make_grid(image, rows, cols):
    step_x = image.shape[1] // cols
    step_y = image.shape[0] // rows

    for i in range(1, cols):
        cv2.line(image, (i * step_x, 0), (i * step_x, image.shape[0]), (255, 255, 255), 2)

    for i in range(1, rows):
        cv2.line(image, (0, i * step_y), (image.shape[1], i * step_y), (255, 255, 255), 2)

    return image

rows = 8
cols = 8

grid_image = transformed_image.copy()
grid_image = make_grid(grid_image, rows, cols)





font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1

for row in range(rows):
    for col in range(cols):
        x = (col * grid_image.shape[1]) // cols + 10
        y = (row * grid_image.shape[0]) // rows + 20



# %%
import numpy as np

def get_grid_intersections(image, rows, cols):
    step_x = image.shape[1] // cols
    step_y = image.shape[0] // rows
    
    intersections = np.zeros((rows + 1, cols + 1, 2), dtype=int)
    
    for i in range(rows + 1):
        for j in range(cols + 1):
            intersections[i, j] = (j * step_x, i * step_y)

    return intersections

rows, cols = 8, 8
intersection_points = get_grid_intersections(transformed_image.copy(), rows, cols)



# %%
import numpy as np
import cv2

original_image = sample_image

inverse_transformation_matrix = cv2.getPerspectiveTransform(target_points.astype(np.float32), chessboard_points.astype(np.float32))
inverse_transformed_image = cv2.warpPerspective(grid_image, inverse_transformation_matrix, (original_image.shape[1], original_image.shape[0]))


def apply_inverse_transformation_to_intersections(intersection_points, inverse_matrix):
    transformed_points = np.zeros_like(intersection_points)

    for i in range(intersection_points.shape[0]):
        for j in range(intersection_points.shape[1]):
            point = np.array([intersection_points[i, j, 0], intersection_points[i, j, 1], 1])  
            transformed_point = inverse_matrix @ point
            transformed_points[i, j] = transformed_point[:2] / transformed_point[2]

    return transformed_points

new_intersection_points = apply_inverse_transformation_to_intersections(intersection_points, inverse_transformation_matrix)

print(new_intersection_points)



# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


marked_image = sample_image.copy()

circle_color = (0, 255, 0)  
circle_radius = 5 
circle_thickness = -1 


for point in new_intersection_points.reshape(-1, 2):  
    x, y = int(point[0]), int(point[1])  
    cv2.circle(marked_image, (x, y), circle_radius, circle_color, circle_thickness)


# %%

# %%
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 5)  
       
        self.fc1 = nn.Linear(16 * 22 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = torch.flatten(x, 1)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return x


model = Net()

# %


import torchvision.transforms.functional as f
from torchvision import transforms

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize((100, 50))])
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

maping = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
def get_image_crops(image_path):
    image = np.array(cv2.imread(image_path))
    image = image.copy()
    dicti = {}
    
        
    mask, green_image = create_green_mask(image)
    mask = mask.copy()
    image_mask = mask  
    threshold = 0.1 

    result_image, corners = Harris_corner_detector(image_mask, threshold)
    best_image = result_image.copy()

    min_distance = 10  
    filtered_corners = filter_chessboard_corners(corners, min_distance)

    sorted_by_y = sorted(filtered_corners, key=lambda corner: corner[1]) 
    lowest_y_points = filter_points(sorted_by_y, distance_threshold)  
    highest_y_points = filter_points(sorted_by_y[::-1], distance_threshold)  

    sorted_by_x = sorted(filtered_corners, key=lambda corner: corner[0])  
    lowest_x_points = filter_points(sorted_by_x, distance_threshold)  
    highest_x_points = filter_points(sorted_by_x[::-1], distance_threshold)  

    line_equations = {}

    line_equations['lowest_y'] = draw_line(lowest_y_points, result_image, (255, 0, 0), 'horizontal')
    line_equations['highest_y'] = draw_line(highest_y_points, result_image, (0, 0, 255), 'horizontal')
    line_equations['lowest_x'] = draw_line(lowest_x_points, result_image, (0, 255, 0), 'vertical')
    line_equations['highest_x'] = draw_line(highest_x_points, result_image, (0, 255, 255), 'vertical')

    intersection_points = find_intersection_points(line_equations)

    intersection_points = intersection_points.values()
    filtered_intersection_points = list(set(
    tuple(point) for point in intersection_points 
    if min_boundary <= point[0] <= max_boundary and min_boundary <= point[1] <= max_boundary
    ))

    points = assign_corners(filtered_intersection_points)
    topleft = points['top_left']
    topright = points['top_right']
    bottomleft = points['bottom_left']
    bottomright = points['bottom_right']

    # print("Top Left:", topleft)
    # print("Top Right:", topright)
    # print("Bottom Left:", bottomleft)
    # print("Bottom Right:", bottomright)

    
    chessboard_boundaries = {
    'top-left': topleft,
    'top-right': topright,
    'bottom-left': bottomleft,
    'bottom-right': bottomright
    }
    chessboard_vector = boundaries_to_vector(chessboard_boundaries)
    target_points = np.array([
        [0, 0],
        [416, 0],
        [416, 416],
        [0, 416]
    ])
    chessboard_points = np.array([chessboard_boundaries[corner] for corner in ['top-left', 'top-right', 'bottom-right', 'bottom-left']])
    transformation_matrix = cv2.getPerspectiveTransform(chessboard_points.astype(np.float32), target_points.astype(np.float32))
    transformed_image = cv2.warpPerspective(best_image, transformation_matrix, (416, 416))

    grid_image = transformed_image.copy()
    grid_image = make_grid(grid_image, rows, cols)
    intersections = get_grid_intersections(transformed_image.copy(), 8, 8)
    inverse_transformation_matrix = cv2.getPerspectiveTransform(target_points.astype(np.float32), chessboard_points.astype(np.float32))
    inverse_transformed_image = cv2.warpPerspective(grid_image, inverse_transformation_matrix, (image.shape[1], image.shape[0]))
    intersections_new = apply_inverse_transformation_to_intersections(intersections, inverse_transformation_matrix)
    for point in intersections_new.reshape(-1, 2):  
            x, y = int(point[0]), int(point[1])  
            cv2.circle(image, (x, y), circle_radius, circle_color, circle_thickness)

    # print(intersections_new)
    image = torch.tensor(image).permute(2, 0, 1)
    # plt.imshow(image.permute(1, 2, 0))
    dicti = {}
    # print(filename)
    for i in range(8):
            for j in range(8):
                # print(i, j)
                # print(intersections_new)
                top = intersections_new[1+i][j][1] - 80
                left = intersections_new[1+i][j][0]
                height = 80
                width = intersections_new[1+i][j+1][0] - left
                if width <= 0:
                    intersections_new = torch.tensor(intersections_new).permute(1, 0, 2)
                    top = intersections_new[1+i][j][1] - 80
                    left = intersections_new[1+i][j][0]
                    height = 80
                    width = intersections_new[1+i][j+1][0] - left
                # print(image.shape)
                # print(top, left, height, width)
                # plt.imshow(image.permute(1, 2, 0))
                # plt.scatter(left, top)
                # plt.scatter(intersections_new[1+i][j][0], intersections_new[1+i][j][1])
                # plt.scatter(intersections_new[1+i][j+1][0], intersections_new[1+i][j+1][1])
                # plt.show()
                # print(top, left, height, width)
                image2 = f.crop(image, top, left, height, width)
                image2 = image2.permute(1, 2, 0).numpy()
                transformed = transforms(image2)
                transformed = np.array(transformed.permute(1, 2, 0))
                dicti[str(maping[i])+str(j+1)] = transformed, 0
    # for key in dicti.keys():
    #     image = torch.tensor(dicti[key][0])
    #     # image = image.permute(2, 0, 1)
    #     plt.imshow(image)
    #     plt.show()
    #     # plt.show()

    return dicti
# print([i.shape for i in get_image_crops(test_im[15]).values()])


        

# %%
model = Net()
model.load_state_dict(torch.load("classification_chess.pth"))

def get_preds_with_constraints(dictionary, threshold=0.88):
    model.eval()
    map2 = {0: 'brak', 1: 'czarny_goniec', 2: 'czarny_krol', 3: 'czarny_kon', 4: 'czarny_pionek', 
            5: 'czarna_królowa', 6: 'czarna_wieża', 7: 'biały_goniec', 8: 'biały_król', 
            9: 'biały_koń', 10: 'biały_pionek', 11: 'biała_królowa', 12: 'biała_wieża'}
    
    expected_counts = {
        4: 8,  
        3: 2,  
        1: 2,  
        6: 2,  
        5: 1,  
        2: 1,  
        10: 8,  
        9: 2,  
        7: 2,  
        12: 2,  
        11: 1,  
        8: 1,  
    }
    
    assigned_counts = {key: 0 for key in expected_counts.keys()}
    
    all_probs = []
    square_keys = list(dictionary.keys())
    with torch.no_grad():
        for idx, key in enumerate(square_keys):
            image = torch.tensor(dictionary[key][0])
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            output = model(image)
            probabilities = F.softmax(output, dim=1).squeeze(0)
            for piece_idx, prob in enumerate(probabilities):
                all_probs.append((prob.item(), idx, piece_idx))  
    

    all_probs.sort(reverse=True, key=lambda x: x[0])  
    

    final_assignments = {key: map2[0] for key in square_keys}  
    assigned_squares = set()
    
    for prob, square_idx, piece_idx in all_probs:
        if prob < threshold or piece_idx == 0:  
            continue
        if assigned_counts[piece_idx] < expected_counts.get(piece_idx, 0) and square_idx not in assigned_squares:
            final_assignments[square_keys[square_idx]] = map2[piece_idx]
            assigned_counts[piece_idx] += 1
            assigned_squares.add(square_idx)
    
    for key in square_keys:
        if key not in final_assignments or final_assignments[key] == map2[0]:
            final_assignments[key] = map2[0]
    
    board_size = int(len(square_keys) ** 0.5)
    for i in range(board_size):
        print([(i, final_assignments[square_keys[j]]) for j in range(i * board_size, (i + 1) * board_size)])

    print(final_assignments)

    return final_assignments

   


get_preds_with_constraints(get_image_crops(r"test\d7887071e972604ddf5940d8eb2702e7_jpg.rf.5f20fe9a6c746d488d6d0478828478cb.jpg"))


def dictionary_to_fen(assignment_dict):
    # Initialize the empty board representation
    board_state = []

    # Iterate over ranks (from 8 to 1)
    for rank in range(8, 0, -1):
        row = []
        # Iterate over files (from 'a' to 'h')
        for file in 'abcdefgh':
            square_key = f"{file}{rank}"  # Form square key
            piece = assignment_dict.get(square_key, 'brak')  # Default to 'brak'
            
            if piece == 'brak':
                row.append('.')  # Use '.' for empty squares
            elif piece == 'czarna_wieża':
                row.append('r')
            elif piece == 'czarny_goniec':
                row.append('b')
            elif piece == 'czarny_krol':
                row.append('k')
            elif piece == 'czarny_kon':
                row.append('n')
            elif piece == 'czarny_pionek':
                row.append('p')
            elif piece == 'czarna_królowa':
                row.append('q')
            elif piece == 'biały_goniec':
                row.append('W')
            elif piece == 'biały_król':
                row.append('K')
            elif piece == 'biały_koń':
                row.append('N')
            elif piece == 'biały_pionek':
                row.append('P')
            elif piece == 'biała_królowa':
                row.append('Q')
            elif piece == 'biała_wieża':
                row.append('R')

        # Convert row to FEN format, counting empty squares
        fen_row = ''
        empty_count = 0

        for square in row:
            if square == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)  # Add count of empty squares
                    empty_count = 0
                fen_row += square
        
        if empty_count > 0:
            fen_row += str(empty_count)  # Add any remaining empty squares
        
        board_state.append(fen_row)  # Append completed row

    # Join the board rows
    fen_string = '/'.join(board_state)

    # Combine into complete FEN
    complete_fen = f"{fen_string}"
    
    return complete_fen

dictionary_to_fen(get_preds_with_constraints(get_image_crops(r"test\d7887071e972604ddf5940d8eb2702e7_jpg.rf.5f20fe9a6c746d488d6d0478828478cb.jpg")))