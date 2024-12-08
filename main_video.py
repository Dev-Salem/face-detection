import cv2
from simple_facerec import SimpleFacerec
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

font_path = "Roboto-Light.ttf"  # Update with your font path
font = ImageFont.truetype(font_path, 24)  # Font size 24 for modern look
text_background_path = "background.png"  # Path to your text background image
text_background = Image.open(text_background_path)

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        bg_width, bg_height = 300, 300  # Adjust size for your text background
        resized_bg = text_background.resize((bg_width, bg_height))
        pil_image.paste(resized_bg, (x1, y1 -20), resized_bg)  # Transparent paste (RGBA)

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)\

        draw.text((x1, y1 + 290), f"{name} (Friend)", font=font, fill=(255, 255, 255))
        draw.text((x1, y1 + 310), "Your best friend", font=font, fill=(255, 255, 255))

        
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()