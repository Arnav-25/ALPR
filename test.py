import cv2
import time
import threading
from class_CNN import NeuralNetwork
from class_PlateDetection import PlateDetector
from utils.average_plate import *
from utils.find_best_quality_images import get_best_images

plateDetector = PlateDetector(minPlateArea=4100, maxPlateArea=15000)

myNetwork = NeuralNetwork(modelFile="model/binary_128_0.50_ver3.pb", labelFile="model/binary_128_0.50_labels_ver2.txt")

list_char_on_plate = [] 
countPlates = 0 
recog_plate = ''
coordinates = (0, 0)
num_frame_without_plates = 0
countPlates_threshold = 11 


def recognized_plate(list_char_on_plate, size):
    global recog_plate

    t0 = time.time()
    plates_value = []
    plates_length = []

    list_char_on_plate = get_best_images(list_char_on_plate, num_img_return=7) 

    for segmented_characters in list_char_on_plate:
        plate, len_plate = myNetwork.label_image_list(segmented_characters[1], size)
        plates_value.append(plate)
        plates_length.append(len_plate)
        
    final_plate = get_average_plate_value(plates_value, plates_length) 
    if len(final_plate) > 7:
        if (final_plate[2] == '8'):
            final_plate = final_plate[:2] + 'B' + final_plate[3:]
        elif (final_plate[2] == '0'):
            final_plate = final_plate[:2] + 'D' + final_plate[3:]
    recog_plate = final_plate  

    print("recognized plate: " + final_plate)
    print("threading time: " + str(time.time() - t0))

cap = cv2.VideoCapture('test_videos/test.MOV') 

if __name__=="__main__":
    while(cap.isOpened()):
        ret, frame = cap.read()
        if (frame is None):
            print("[INFO] End of Video")
            break

        _frame = cv2.resize(frame, (960, 540)) 
        frame_height, frame_width = frame.shape[:2]
        _frame_height, _frame_width = _frame.shape[:2]
        cropped_frame = frame[int(frame_height*0.3):frame_height, 0:int(frame_width*0.8)] 
        cv2.rectangle(_frame, (0, int(_frame_height*0.3)), (int(_frame_width*0.8), _frame_height), (255, 0, 0), 2)

        cv2.rectangle(_frame, (0, 0), (190, 40), (0, 0, 0), -1)
        cv2.putText(_frame, recog_plate, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('video', _frame)
        
        possible_plates = plateDetector.find_possible_plates(cropped_frame)
        if possible_plates is not None:
            num_frame_without_plates = 0
            distance = tracking(coordinates, plateDetector.corresponding_area[0]) 
            coordinates = plateDetector.corresponding_area[0]
            if (distance < 100):
                if(countPlates < countPlates_threshold):
                    cv2.imshow('Plate', possible_plates[0])
                    temp = []
                    temp.append(possible_plates[0])
                    temp.append(plateDetector.char_on_plate[0]) 

                    list_char_on_plate.append(temp)
                    countPlates += 1
                elif(countPlates == countPlates_threshold):
                    threading.Thread(target=recognized_plate, args=(list_char_on_plate, 128)).start()
                    countPlates += 1
            else:
                countPlates = 0
                list_char_on_plate = []

        if (possible_plates == None):
            num_frame_without_plates += 1
            if (countPlates <= countPlates_threshold and countPlates > 0 and num_frame_without_plates > 5):
                threading.Thread(target=recognized_plate, args=(list_char_on_plate, 128)).start()
                countPlates = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()