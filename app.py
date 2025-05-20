import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
from create_answers import create_answers
import qrcode






def pdf_to_image(
    pdf_path, dpi=300,
    path='converted_sheets/', save=False):
    
    '''Converts PDF to PNG and save with the PDF name'''
    
    os.makedirs(path, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path='poppler/poppler-24.08.0/Library/bin')
    
    # If the PDF contains more than one page
    if len(images) > 1:
        path = f'{path}{pdf_path.split('/')[-1][:-4]}' # get the PDF name
        for i, img in enumerate(images, start=1):
            cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            img.save(f'{path}{i}.png')
            print('sad')
    else:
        cv2.resize(images[0], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        images[0].save(f"{path}{pdf_path.split('.')[0][-4:]}.png")
        print('saved')
    

    
def get_details(image_path):
    image = cv2.imread(image_path, 0)
    
    cv2.imwrite("contrast_enhanced.png", image)  # or equalized

    # Detect 
    qr_detector = cv2.QRCodeDetector()
    data, _, _ = qr_detector.detectAndDecode(image)
    
    return data




  
# Convert all scanned sheets to PNG format
# for sheet in os.listdir('answered_sheets'):
#     if sheet.endswith('.pdf'):
#         pdf_to_image(f'answered_sheets/{sheet}')


# # Detect QR Code and get details
# for images in os.listdir('converted_sheets'):
#     data = get_details(f'converted_sheets/{images}')
#     print(data)


lists = [1, 2,3, 4, 5, 5]

print([*lists])

