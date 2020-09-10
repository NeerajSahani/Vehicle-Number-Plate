
import os, cv2, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

os.chdir('E:/Python/Notes/Data Science/My Notes/Number Plate')
letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
haarCascade = cv2.CascadeClassifier('./NumberPlateHaarCascade.xml')



def get_model():
    with open('./recognizer/model.json') as file:
        json = file.read()
        model = model_from_json(json)
        model.load_weights('./recognizer/weights.h5')
        print("Done")
        return model
    
model = get_model()

def extract_plate(img): # the function detects and perfors blurring on the number plate.
    plate_img = img.copy()
    #Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier('./NumberPlateHaarCascade.xml')
    
	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)
    if len(plate_rect)>0:
        for (x,y,w,h) in plate_rect:
            a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning
            plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
            # finally representing the detected contours by drawing rectangles around the edges.
            cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
    
        return plate_img, plate # returning the processed image.
    return img, []



def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)

    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
    
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res



def segment_characters(image) :

    # Preprocess cropped license plate image
    img = cv2.resize(image, (333, 75))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, (3,3))
    img_dilate = cv2.dilate(img_erode, (3,3))

    LP_WIDTH = img_dilate.shape[0]
    LP_HEIGHT = img_dilate.shape[1]

    # Make borders white
    img_dilate[0:3,:] = 255
    img_dilate[:,0:3] = 255
    img_dilate[72:75,:] = 255
    img_dilate[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_dilate)

    return char_list

def fix_dimension(img):
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img

def predict(path):  #pass path or numpy image
    if 'str' in str(type(path)):
        img = cv2.imread(path)
    else:
        img=path
    img, plate = extract_plate(img)
    try:
        cv2.imshow('Plate', plate)
    except:
    
    if len(plate) !=0:
        try:
            char_list = segment_characters(plate)
            out = ''
            for ch in char_list:                
                ch_resized=cv2.resize(ch, (28, 28))
                ch_fixed = fix_dimension(ch_resized)
                
              #  plt.imshow(ch_fixed)
                ch_fixed.shape = [1, 28, 28, 3]
                ot = model.predict_classes(ch_fixed)[0]
                out += letters[ot]
            return out
                    
        except Exception as e:
            print(e)
#predict()

def realTime(path=0):   #default web cam
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    while ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        plates = haarCascade.detectMultiScale(gray, 1.3, 7)
        if len(plates) > 0:
            for x, y, w, h in plates:
                a,b = (int(0.02*frame.shape[0]), int(0.025*frame.shape[1]))
                plate = frame[y+a:y+h-a, x+b:x+w-b, :]
                frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (51,51,255), 3)
                try:
                    char_list = segment_characters(plate)
                    out = ''
                    for ch in char_list:                
                        ch_resized=cv2.resize(ch, (28, 28))
                        ch_fixed = fix_dimension(ch_resized)
                        ch_fixed.shape = [1, 28, 28, 3]
                        ot = model.predict_classes(ch_fixed)[0]
                        out += letters[ot]
                    frame = cv2.putText(frame, out, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                    print(out)
                except Exception as e:
                    print(e)
        cv2.imshow('Video', frame)
        ret, frame = cap.read()
        if cv2.waitKey(5) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
    
        
            

path='C:/Users/Asus/Downloads/Hikvision License Plate Recognition (LPR) Camera Demo Video.mp4'
realTime(path=path)            
