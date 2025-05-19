import qrcode
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import fitz 
import random


def generate_answer_sheet(data, pdf_template='docs/answer_sheet_temp.pdf'):
    # Generate QR Code
    qr_img = qrcode.make(data)
    qr_path = f"answer_sheets/qr_images/qr_{data.get('roll')}.png"
    qr_img.save(qr_path) # pyright: ignore[reportArgumentType] # type: ignore
    
    # Load it and resize
    qr_resized = Image.open(qr_path).resize((int(2.9*cm), int(2.9*cm)))
    
    output_pdf_path = f'answer_sheets/{data.get("roll")}.pdf'
    
    # Load the answer sheet template
    doc = fitz.open(pdf_template)
    page = doc[0]  # First page
    
    page_width = page.rect.width
    qr_width = qr_resized.width
    
    x = (page_width - qr_width) / 2  # center horizontally
    y = 25  # distance from the top

    # Insert QR code image to PDF
    page.insert_image(fitz.Rect(x, y, x + qr_width, y + qr_width), # type: ignore
                    filename=qr_path)

    # Save new PDF 
    doc.save(output_pdf_path)
    doc.close()
    
    

if __name__ == '__main__':   
    for name in ['Riyas', 'Shihan', 'Safu', 'Nachu', 'Muthu', 'Resu', 'Pazham', 'Ansari']:
        roll_no = random.randint(3000, 4000)
        data = {
            "name": name,
            "roll": roll_no
        }
        generate_answer_sheet(data)