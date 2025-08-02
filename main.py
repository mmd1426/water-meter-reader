# Import libraries
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# Create round border in the input image
def add_round_border(
    image, border_color=(232, 232, 232), border_radius=30, border_width=3
):  
    image = image.convert("RGBA")
    # Create an out mask and an in mask
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [0, 0, image.size[0], image.size[1]], radius=border_radius, fill=255
    )
    mask_in = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask_in)
    draw.rounded_rectangle(
        [
            border_width,
            border_width,
            image.size[0] - border_width,
            image.size[1] - border_width,
        ],
        radius=border_radius - border_width,
        fill=255,
    )

    border_image = Image.new("RGBA", image.size, color=border_color)
    new_image = Image.new("RGBA", image.size, color=(220,220,65))
    # Add the border by pasting the border images onto the new image
    new_image.paste(border_image, mask=mask)
    new_image.paste(image, mask=mask_in)
    return new_image


# Load model
model = YOLO(r'/src/models/model.pt')

# Classes
labels = ['0','1','2','3','4','5','6','7','8','9']

# Reading input image
img = cv2.imread(r'/src/test/input.png')

# Resizing input image
img = cv2.resize(img,(720,525))

# Create lists
boxes = []
labels_list = []

# Load background
cap = cv2.VideoCapture(r'/src/backgrounds/background.png')

while True:

    # Reading frame
    ret,frame = cap.read()

    # Resizinf frame
    frame = cv2.resize(frame,(900,700))

    # Predicting input image
    results = model.predict(img)[0]

    # Getting the coordinates of the predicted boxes
    for r in results.boxes.data.tolist():
        x1 , y1 , x2 , y2 , score , class_id = r
        x1 , y1 , x2 , y2 , class_id = int(x1) , int(y1) , int(x2) , int(y2) , int(class_id)


        boxes.append([x1,y1,x2,y2,class_id])

    # Sorting boxes list
    boxes = sorted(boxes)

    # Actions on the frame
    for i in range(len(boxes)):
        cv2.rectangle(img,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,0,255),2)
        id = boxes[i][4]
        label = labels[id]
        labels_list.append(label)
        cv2.putText(img,label,(boxes[i][0],boxes[i][1]-5),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
    
    # Actions on the input image
    img = Image.fromarray(img)
    img = add_round_border(img,(220,220,65),30,1)
    img = np.asarray(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    frame[25:550,90:810] = img

    # Placing fonts on the input image
    frame = Image.fromarray(frame)
    font = ImageFont.truetype(r'/src/fonts/ARIALBD.TTF',size=50)
    drawing_on_img = ImageDraw.Draw(frame)
    j = 280
    for i in range(len(labels_list)-1):
        drawing_on_img.text((j,597),labels_list[i],font=font,fill='white')
        j += 59
    drawing_on_img.text((j+16,597),labels_list[len(labels_list)-1],font=font,fill='blue')
    frame = np.asarray(frame)


    cv2.imshow('WaterMeter-Reader',frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()