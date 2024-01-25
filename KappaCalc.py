
# The given program has been adjusted to include the calculation of curvature at each point
# along the contour and to display the average curvature value.

import cv2
import numpy as np
from scipy.interpolate import interp1d

# Function to calculate curvature at each point
def calculate_curvature(pts, idx):
    if idx == 0 or idx == len(pts) - 1:
        return 0  # Cannot compute curvature for the first and last points
    pt1 = np.array(pts[idx - 1])
    pt2 = np.array(pts[idx])
    pt3 = np.array(pts[idx + 1])

    # Set z-coordinate as 0 for the vectors (2D to 3D conversion)
    pt1 = np.append(pt1, 0)
    pt2 = np.append(pt2, 0)
    pt3 = np.append(pt3, 0)

    # Calculate the first and second derivative (tangent and sec_derivative)
    tangent = pt2 - pt1
    sec_derivative = pt3 - 2 * pt2 + pt1
    
    # Calculate the curvature (kappa)
    # Kappa = |r'(t) x r''(t)| / |r'(t)|^3
    numerator = np.linalg.norm(np.cross(tangent, sec_derivative))
    denominator = np.linalg.norm(tangent) ** 3 if np.linalg.norm(tangent) != 0 else 1e-5
    curvature = numerator / denominator
    return curvature

# Load and read image in color
image = cv2.imread('images/Qiyana_cont.png')

# Check if image is loaded
if image is None:
    print("Error: Image not found")
    exit()

# Check if the image has an alpha channel for transparency
if image.shape[2] == 4:
    # Use the alpha channel as a mask
    _, mask = cv2.threshold(image[:, :, 3], 1, 255, cv2.THRESH_BINARY)
else:
    # Convert to grayscale and threshold to create a mask if no alpha channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Check and resize the image if it's larger than the maximum dimensions
height, width = image.shape[:2]
max_height = 600
max_width = 800

# Only resize if the image is larger than the maximum dimensions
if max_height < height or max_width < width:
    # Calculate the ratio of the height and construct the dimensions
    scaling_factor = min(max_height / float(height), max_width / float(width))
    new_dimensions = (int(width * scaling_factor), int(height * scaling_factor))
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, new_dimensions, interpolation=cv2.INTER_AREA)

# Find contours from the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the subject, since background is empty
main_subject_contour = max(contours, key=cv2.contourArea)

# Draw the largest contour onto the resized image
cv2.drawContours(image, [main_subject_contour], -1, (0, 255, 0), 2)

# Calculate the cumulative arc length for each point on the contour
arc_lengths = np.cumsum(np.sqrt(np.sum(np.diff(main_subject_contour[:, 0, :], axis=0)**2, axis=1)))
arc_lengths = np.insert(arc_lengths, 0, 0)  # Insert a 0 at the beginning for the start point
total_arc_length = arc_lengths[-1]

# Normalize the arc lengths so that they range from 0 to 1
normalized_arc_lengths = arc_lengths / total_arc_length

# Define the interpolation functions for x and y coordinates
interp_x = interp1d(normalized_arc_lengths, main_subject_contour[:, 0, 0], kind='linear')
interp_y = interp1d(normalized_arc_lengths, main_subject_contour[:, 0, 1], kind='linear')

# Generate evenly spaced points along the contour
num_points = 500
evenly_spaced_lengths = np.linspace(0, 1, num_points)
evenly_spaced_points_x = interp_x(evenly_spaced_lengths)
evenly_spaced_points_y = interp_y(evenly_spaced_lengths)
evenly_spaced_points = np.vstack((evenly_spaced_points_x, evenly_spaced_points_y)).T

# Calculate curvature at each point and store in a list
curvatures = [calculate_curvature(evenly_spaced_points, i) for i in range(num_points)]

# Average the curvature values
average_curvature = np.mean(curvatures)

# Display the result
print(f"Average Curvature: {average_curvature}")

# Draw the evenly spaced points onto the image
for point in evenly_spaced_points:
    cv2.circle(image, tuple(point.astype(int)), 1, (0, 0, 255), -1)

# Display the image with the evenly spaced points
cv2.imshow('Evenly Spaced Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
