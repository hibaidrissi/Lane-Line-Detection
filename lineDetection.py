import matplotlib.pylab as plt
import cv2
import numpy as np

# Marquer que la route
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    


# center_exclusion_width_left=0.0, center_exclusion_width_right=0.0 pour les autres images
# center_exclusion_width_left=0.0, center_exclusion_width_right=0.3 pour challenge_video.mp4

def calculate_lane_lines(img, lines, left_list, right_list, center_exclusion_width_left=0.0, center_exclusion_width_right=0.0):
    if lines is None:
        lines = [] 

    center = img.shape[1] / 2 
    exclusion_left = center - (center * center_exclusion_width_left)
    exclusion_right = center + (center * center_exclusion_width_right)

    left_lines = []
    right_lines = [] 

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:  # pas de divisions par 0
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)

            if (x1 > exclusion_left and x1 < exclusion_right) or (x2 > exclusion_left and x2 < exclusion_right):
                continue  

            # Filtrer les ignes horizontales
            if abs(slope) < 0.5:
                continue

            if slope < 0: 
                left_lines.append((slope, intercept))
            else:  
                right_lines.append((slope, intercept))

    def average_slope_intercept(lines):
        if lines:
            slope, intercept = np.mean(lines, axis=0)
            y1 = img.shape[0]
            y2 = int(y1 * 0.6)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1, x2, y2)
        else:
            return None

    # Utiliser l'historique des lignes si les lignes ne sont pas détéctées
    left_line = average_slope_intercept(left_lines) if left_lines else left_list[-1] if left_list else None
    right_line = average_slope_intercept(right_lines) if right_lines else right_list[-1] if right_list else None

    return left_line, right_line


 # Dessiner les lignes sur l'image
def draw_the_lines(img, left_line, right_line, left_list, right_list):
    line_img = np.zeros_like(img)
    
    # utiliser l'historique
    if left_line is None and left_list:
        left_line = left_list[-1]
    if right_line is None and right_list:
        right_line = right_list[-1]

    poly_points = []

    # Créer un polygone si on a les 2 lignes
    if left_line is not None and right_line is not None:
        # ligne gauche
        poly_points.append([left_line[0], left_line[1]]) 
        poly_points.append([left_line[2], left_line[3]])  
        
        # ligne droite
        poly_points.append([right_line[2], right_line[3]])  
        poly_points.append([right_line[0], right_line[1]])  

        poly_points = np.array([poly_points], dtype=np.int32)
        cv2.fillPoly(line_img, poly_points, (0, 255, 0)) 

    for line in [left_line, right_line]:
        if line is not None:
            cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 5)  # Thick red line

    img_with_lines = cv2.addWeighted(img, 0.8, line_img, 1, 0.0)
    return img_with_lines

# Calculer la direction et la distance
def determine_heading_and_distance(img, left_line, right_line, left_list, right_list, draw_text=True, lane_width_meters=3.5, image_width_pixels=1280):
    height, width = img.shape[:2]
    center_of_image = width / 2
    direction = ""
    distance_from_center_meters = None

    if left_line is None and left_list:
        left_line = left_list[-1]
    if right_line is None and right_list:
        right_line = right_list[-1]

    if left_line is not None and right_line is not None:

        def line_midpoint(line):
            x1, y1, x2, y2 = line
            return (x2, y2)  

        if left_line is not None and right_line is not None:
            left_mid = line_midpoint(left_line)
            right_mid = line_midpoint(right_line)

            lane_midpoint_pixels = (left_mid[0] + right_mid[0]) / 2

            if lane_midpoint_pixels + 5 < center_of_image:
                direction = "Left"
            elif lane_midpoint_pixels - 5 > center_of_image:
                direction = "Right"
            else:
                direction = "Straight"

            distance_from_center_pixels = center_of_image - lane_midpoint_pixels

            pixels_per_meter = image_width_pixels / lane_width_meters
            distance_from_center_meters = distance_from_center_pixels / pixels_per_meter

            if draw_text:
                cv2.putText(img, f"Heading: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f"Offset: {np.abs(distance_from_center_meters):.2f}m to the {direction}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return img, direction, distance_from_center_meters


def process(image, left_list, right_list):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [(0, height), (width/2, height/2),(width, height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 200, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image, rho=1, theta=np.pi/180, threshold=80, lines=np.array([]), minLineLength=100, maxLineGap=100)
    #CHANGER LES VALEURS !
    # center_exclusion_width_left=0.0, center_exclusion_width_right=0.0 pour les autres videos
    # center_exclusion_width_left=0.0, center_exclusion_width_right=0.3 pour challenge_video.mp4
    left_line, right_line = calculate_lane_lines(image, lines,left_list, right_list, center_exclusion_width_left=0.0, center_exclusion_width_right=0.0)
    if left_line is not None : 
        left_list.append(left_line)
    if right_line is not None : 
        right_list.append(right_line)

    image_with_lines = draw_the_lines(image, left_line, right_line, left_list, right_list)
    image_with_lines, direction, dist = determine_heading_and_distance(image_with_lines, left_line, right_line,left_list, right_list, draw_text=True)

    return image_with_lines, direction, dist




cap = cv2.VideoCapture('Images/project_video.mp4')
#cap = cv2.VideoCapture('Images/challenge_video.mp4')
#cap = cv2.VideoCapture('Images/harder_challenge_video.mp4')

left_list = []
right_list = []

while cap.isOpened():
    
    ret, frame = cap.read()
    frame, direction, dist = process(frame, left_list, right_list)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
