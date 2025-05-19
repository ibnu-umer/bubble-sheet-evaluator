import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
from create_answers import create_answers
import qrcode
# from pyzbar.pyzbar import decode



def pdf_to_image(pdf_path, dpi=300):
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path='poppler/poppler-24.08.0/Library/bin')
    img_path = "results/converted_page.png"
    images[0].save(img_path, 'PNG')
    return img_path


def detect_border_shapes(image_path):
    '''Detect and crop image'''
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150) # Apply edge detection
    cv2.imwrite('results/edged.png', edged)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    h, w = gray.shape
    corner_centers = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        
        if w_box < 100 or w_box > 130 and 110 < h_box or h_box > 140:
            continue  # skip noise and huge objects
        
        # Only keep contours near the image corners
        if (x < w * 0.2 and y < h * 0.2) or \
           (x + w_box > w * 0.8 and y < h * 0.2) or \
           (x < w * 0.2 and y + h_box > h * 0.8) or \
           (x + w_box > w * 0.8 and y + h_box > h * 0.8):
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 3)
            
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            # Find the vertex closest to the contour's bounding box corner
            if len(approx) >= 3:
                # Choose point with the minimum sum of (x + y) if it's top-left,
                # Or customize this logic per-corner
                if x < w * 0.2 and y < h * 0.2:  # Top-left
                    corner = min(approx, key=lambda pt: pt[0][0] + pt[0][1])
                elif x + w_box > w * 0.8 and y < h * 0.2:  # Top-right
                    corner = min(approx, key=lambda pt: -pt[0][0] + pt[0][1])
                elif x < w * 0.2 and y + h_box > h * 0.8:  # Bottom-left
                    corner = min(approx, key=lambda pt: pt[0][0] - pt[0][1])
                else:  # Bottom-right
                    corner = max(approx, key=lambda pt: pt[0][0] + pt[0][1])

                # Check if the contuor looks like a L
                for i in range(len(approx)):
                    pt1 = approx[i - 1][0]
                    pt2 = approx[i][0]
                    pt3 = approx[(i + 1) % len(approx)][0]

                    # Check the angle of the points
                    ang = angle(pt1, pt2, pt3)
                    
                    # If it is a corner
                    if 80 <= ang <= 100:
                        corner_point = tuple(corner[0])
                        corner_centers.append(corner_point)
                        cv2.circle(result, corner_point, 10, (255, 0, 255), -1)  # Magenta dot at intersection
    
    corner_centers = set(corner_centers)  # Convert to set to remove duplicates
    cv2.imwrite('results/det_cor.png', result)
    
    # CROPPING
    # Proceed only if all 4 corners are found
    if len(corner_centers) == 4:
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        sorted_by_y = sorted(corner_centers, key=lambda pt: pt[1])
        top_two = sorted(sorted_by_y[:2], key=lambda pt: pt[0])   # left to right
        bottom_two = sorted(sorted_by_y[2:], key=lambda pt: pt[0])
        pts_src = np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]], dtype="float32")

        # Define output size (A4 at 300dpi)
        width, height = 2480, 3508
        pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

        # Warp perspective
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        cv2.imwrite("results/cropped_sheet.png", warped)
        return warped

    else:
        print(f"Only {len(corner_centers)} corner(s) detected. Cannot perform crop.")
    
# Calculate angle between consecutive segments
def angle(pt1, pt2, pt3):
    v1 = pt1 - pt2
    v2 = pt3 - pt2
    angle = np.arccos(
        np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    )
    return np.degrees(angle)
      

def detect_blobs(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)

    circles = cv2.HoughCircles(blurred, 
                        cv2.HOUGH_GRADIENT,
                        dp=1,
                        minDist=30,
                        param1=50,
                        param2=20,
                        minRadius=35,
                        maxRadius=40)
    
    circles = np.round(circles[0, :]).astype("int")
    for i, (x, y, r) in enumerate(circles):
        
        roi = thresh[y - r:y + r, x - r:x + r] # Extract ROI (Region of Interest)
        mean_intensity = cv2.mean(roi)[0]

        # Draw outer circle (outline)
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)

        if mean_intensity > 70:  
            cv2.circle(image, (x, y), r - 2, (0, 0, 255), -1)

    if len(circles[0]) <= 160:  # 40 x 4 = 160 
        # circles = np.round(circles[0, :]).astype("int")
        # for i, (x, y, r) in enumerate(circles):
            
        #     roi = thresh[y - r:y + r, x - r:x + r] # Extract ROI (Region of Interest)
        #     mean_intensity = cv2.mean(roi)[0]

        #     # Draw outer circle (outline)
        #     cv2.circle(image, (x, y), r, (0, 255, 0), 2)

        #     if mean_intensity > 70:  
        #         cv2.circle(image, (x, y), r - 2, (0, 0, 255), -1)
                
        cv2.imwrite('marked.png', image)
        return circles, thresh
    
    else:
        print(f'{len(circles)} circles found.')
        cv2.imwrite('false_marked.png', image)
        return None, None


def group_by_rows(circles, thresh, y_thresh=15):
    rows = []
    for c in sorted(circles, key=lambda x: x[1]):  # sort by Y
        placed = False
        for row in rows:
            if abs(row[0][1] - c[1]) < y_thresh:  # compare Y with 1st bubble in row
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])

    # Sort circles by X - from left to right
    for idx, row in enumerate(rows):
        row.sort(key=lambda x: x[0])  # sort by X
        
    # Split the row in 4, 4. Coz it contains options of two question. eg: 1 and 20, 2 and 21
    question_map = {}
    options = ['A', 'B', 'C', 'D']
    for qn, row in enumerate(rows, start=1):
        left_row = row[:4]
        right_row = row[4:]

        # First 4 bubbles
        for j, (x, y, r) in enumerate(left_row):
            roi = thresh[y - r:y + r, x - r:x + r] # Extract ROI (Region of Interest)
            mean_intensity = cv2.mean(roi)[0]
            if mean_intensity < 70: # If colored
                if qn in question_map:  # If multiple choices, do not select
                    del question_map[qn]  # Delete the saved option
                    break  # Break the loop to do not iterate for another choice
                question_map[qn] = [options[j], x, y, r, mean_intensity]  
        
        # Last 4 bubbles
        for j, (x, y, r) in enumerate(right_row):
            roi = thresh[y - r:y + r, x - r:x + r] # Extract ROI (Region of Interest)
            mean_intensity = cv2.mean(roi)[0]
            if mean_intensity < 70: # If colored
                question_map[qn + 20] = [options[j], x, y, r, mean_intensity]
                
    
    # print('qmap', question_map)
    answers = {}
    for qn, values in question_map.items():
        answers[qn] = values[0]
    
    
    
    image = cv2.imread('results/cropped_sheet.png')
    for qn, value_list in question_map.items():
        cv2.putText(
                image, str(qn), (value_list[1]-100, value_list[2]),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 3, cv2.LINE_AA
                )
        
        cv2.putText(
            image, str(value_list[0]), (value_list[1], value_list[2]),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 0, 0), 3, cv2.LINE_AA
        )
    cv2.imwrite('result.png', image)
    
    
    
    return answers
        
        
def get_details(image_path):
    image = cv2.imread(image_path)
    
    qr_detector = cv2.QRCodeDetector()
    data, _, _ = qr_detector.detectAndDecode(image)
    
    return data
        
        
        
def evaluate(answers, data):
    correct_answers = create_answers(40)
    print(answers)
    
    total_mark = 0
    for qn, answer in answers.items():
        crr = correct_answers.get(qn)
        if answer == crr:
            total_mark += 1
    
    print(f'Details : {data}')
    print('Total mark :', total_mark)      
        
    
            
        







if __name__ == '__main__':
    file_path = 'results/converted_page.png'
    if file_path.split('.')[-1] == 'pdf':
        image_path = pdf_to_image(file_path)
    else:
        image_path = file_path
    data = get_details(file_path)
    croped_image = detect_border_shapes(image_path)
    circles, thresh = detect_blobs(croped_image)
    answers = group_by_rows(circles, thresh)

    evaluate(answers, data)