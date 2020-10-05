#import all the necessary libraries before execution of the program
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os.path
from os import path

def train_images():
    files = glob.glob("Training Images\\*")
    training_images = files[:int(len(files))-1]
    train_descriptors = []

    for item in training_images:
        img = cv2.imread(item)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        img_keypoints , img_descriptor = orb.detectAndCompute(img_gray, None)
        train_descriptors.append(img_descriptor)

    with open('train_descriptors', 'wb') as f:
        pickle.dump(train_descriptors, f)

#---------------------------------------------------------------------------------------------------#

if path.exists("train_descriptors") == True:
    print("\nTraining_descriptors Found !!!") 
else:
    print("\nNo Training_descriptors Found !!!")
    print("\nTraining Images")
    train_images()
    print("\nCreated descriptors and sucessfully saved !!!")

with open('train_descriptors', 'rb') as f:
    temp = pickle.load(f)
    train_descriptors = np.array(temp)
    

orb = cv2.ORB_create()

#Initialize Camera
path='E:/project/Python123/Similarity Measure/Space Rebooters/Testing Image' # you need to change according to your file location
camera = cv2.VideoCapture(0)
cv2.namedWindow("test")

while True:
    return_value, image = camera.read()
    if not return_value:
        print("failed to grab frame")
        break
    cv2.imshow("test", image)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        cv2.imwrite(os.path.join(path,'image'+'.jpg'), image)
        
        break
camera.release()

cv2.destroyAllWindows()

test_image = cv2.imread("Testing Image\\image.jpg")
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
test_keypoints, test_descriptors = orb.detectAndCompute(test_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
x=[]
for i in train_descriptors:
    for j in i:
        x.append(j)
train_descriptors = np.array(x)

matches = bf.match(train_descriptors, test_descriptors)  
matches = sorted(matches, key = lambda x : x.distance) 

print("\nNumber of Keypoints Detected in Test Image : ", len(test_descriptors))
print("\nNumber of Keypoints Matched  in Test Image : ", len(matches))
accuracy = len(matches)/len(test_descriptors)
accuracy = accuracy * 100
print("Accuracy",accuracy)
if(accuracy > 60):
    files = glob.glob("Training Images\\*")
    cv2.imwrite("Training Images\\image%i.jpg"%(len(files)+1),test_image) 
    train_images()
    print('YES!! IMAGE IS FOUND IN THE DATABASE')
else:
    print('NO IMAGE IS FOUND IN THE DATABASE')
print('\n')


# Initialize lists
matched_keypoints = []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images
    index = mat.trainIdx

    matched_keypoints.append(test_keypoints[index])

unmatched_points = []
for i in test_keypoints:
    if i not in matched_keypoints:
        unmatched_points.append(i)

img_1 = cv2.drawKeypoints(test_image,matched_keypoints,test_image,(0,0,255))
img_2 = cv2.drawKeypoints(test_image,unmatched_points,test_image,(255,0,0))
cv2.imshow("Image", img_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
os.remove("E:/project/Python123/Similarity Measure/Space Rebooters/train_descriptors")  # you need to change according to your file location
