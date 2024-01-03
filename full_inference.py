#import modules

import cv2

import numpy as np

# from pygame import mixer

import threading

import serial

from time import sleep



import torch

from torch import nn

from torchvision import transforms, models



from PIL import Image



#set random seed for reproducibility

seed = 42



Alarm_Status = False

Fire_Reported = False



fire_counter=0



gsm = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.5)

gsm.flush()



device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)



weights = models.MobileNet_V3_Small_Weights.DEFAULT





#get the data transformations used to create our pretrained weights

auto_transforms = weights.transforms()





model_tl = models.mobilenet_v3_small(weights=weights).to(device)





for param in model_tl.features.parameters():

    param.requires_grad = False



#set the manual seeds

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)



#redefine classifier layer

model_tl.classifier = torch.nn.Sequential(

    nn.Linear(in_features=576, out_features=1024, bias=True),

    nn.Hardswish(),

    torch.nn.Dropout(p=0.2, inplace=True),

    torch.nn.Linear(in_features=1024, out_features=1, bias=True)).to(device)



# def play_alarm_sound_function():

	# while True:

	    # mixer.init()

	    # mixer.music.load("alarm-sound.mp3")

	    # mixer.music.set_volume(0.7)

	    # mixer.music.play()

        



def sendSms(msg, mobile_number):

    print("Sending SMS\n")

    gsm.write(b'AT+CMGF=1\r\n')

    sleep(0.5)

    gsm.write(b'AT+CMGS="')

    gsm.write(mobile_number.encode())

    gsm.write(b'"\r\n')

    sleep(0.5)

    data = msg

    gsm.write(data.encode())

    gsm.write(b'\x1A')

    sleep(15)



def makeCall(phone_number):

    sleep(5)

    print(f"Making a call to {phone_number}\n")

    gsm.write(b'ATD')  # Dial command

    gsm.write(phone_number.encode())  # Phone number to call

    gsm.write(b';\r\n')  # End the command with semicolon and newline

    sleep(10)  # Sleep for 10 seconds (adjust as needed)

    gsm.write(b'ATH0\r\n')  # Hang up the call



#define function to predict custom images

def pred_image(model, image_path, transform, device: torch.device=device):



    img = Image.open(image_path)



    image_transform = transform



    model.to(device)

    model.eval()

    with torch.inference_mode():



        transformed_image = image_transform(img).unsqueeze(dim=0)

        target_image_pred = model(transformed_image.to(device))



        target_image_pred_probs = torch.sigmoid(target_image_pred)

        target_image_pred_label = torch.round(target_image_pred_probs)

    

    return target_image_pred_label[0].item()







def Load_model(model, filename):

    model.load_state_dict(torch.load(filename))



Load_model(model_tl, 'best_model_tl.pth')



video = cv2.VideoCapture("fire.mp4") # If you want to use webcam use Index like 0,1.





while True:

    (grabbed, frame) = video.read()

    if not grabbed:

        break

    

    frame=cv2.resize(frame, (720, 640),

               interpolation = cv2.INTER_LINEAR)

    cv2.imwrite("image.png", frame)



    Fire_Reported=bool(pred_image(model_tl, "image.png", auto_transforms, device))



    print(Fire_Reported)



    cv2.imshow("output", frame)



    if Fire_Reported==True:

        fire_counter+=1

        if Alarm_Status == False and fire_counter>10:

            # threading.Thread(target=play_alarm_sound_function).start()

            threading.Thread(target=sendSms("Fire Alert", "7020036643")).start()

            threading.Thread(target=makeCall("7020036643")).start()

            

            Alarm_Status = True

    else:

        fire_counter=0

            



    if cv2.waitKey(1) & 0xFF == ord(' '):

        break



cv2.destroyAllWindows()

video.release()

