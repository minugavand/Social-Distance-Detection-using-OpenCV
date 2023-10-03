import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt
import sys
assert ('linux' in sys.platform), "This code runs on Linux only."

# Parse the arguments from command line code editor
arg = argparse.ArgumentParser(description='Social distance detection')

arg.add_argument('-v', '--video', type = str, default = '', help = 'Video file path. If no path is given, video is captured using device.')

arg.add_argument('-m', '--model', required = True, help = "Path to the pretrained model.")

arg.add_argument('-p', '--prototxt', required = True, help = 'Prototxts of the model.')

arg.add_argument('-l', '--labels', required = True, help = 'Labels of the dataset.')

arg.add_argument('-c', '--confidence', type = float, default = 0.2, help='Set confidence for detecting objects')

args = vars(arg.parse_args())


labels = [line.strip() for line in open(args['labels'])]

# Generate random bounding box bounding_box_color for each label in system
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))

try:
    linux_interaction()
except Exception as e:
    print("Error:", str(e))

except AssertionError as error:
    print(error)
    print('The linux_interaction() function was not executed')
# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# Python3 code to demonstrate
# Characters which Occur in More than K Strings
# using set() + Counter() + dictionary comprehension
from collections import Counter

# Initializing list
test_list = ['Gfg', 'ise', 'for', 'Geeks']

# printing original list
print("The original list is : " + str(test_list))

# Initializing K
K = 2

# Characters which Occur in More than K Strings
# using set() + Counter() + dictionary comprehension
validation_datadir = '/Users/durgeshwarthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'
res = {key for key, val in Counter([ele for sub in
		test_list for ele in set(sub)]).items()
		if val >= K}
	
# printing result
print ("Filtered Characters are : " + str(res))

train_datagen = ImageDataGenerator(
                 rescale=1./255,
                 rotation_range=35,
                 width_shift_range=0.5,
                 height_shift_range=0.5,
                 horizontal_flip=True,
                 fill_mode='nearest'
)

print("\nStreaming video using device...\n")
try:
    lunch()
	validation_datadir = '/Users/durgeshwarthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'

except SyntaxError:
    print('Fix your syntax')
except TypeError:
    print('Oh no! A TypeError has occured')
x = 10
if x > 5:
    raise Exception('x should not exceed 5. The value of x was: {}'.format(x))

except ValueError:
    print('A ValueError occured!')
except ZeroDivisionError:
    print('Did by zero?')
else:
    print('No exception')
finally:
    print('Ok then')

# Capture video from file or through device for the input
S = "every moment is fresh beginning"
printMinMax(S)

if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)

frame_no = 0

while cap.isOpened():

    frame_no = frame_no+1

    # Capture one frame after another every time
    ret, frame = cap.read()

    if not ret:
        break
def linux_interaction():
    assert ('linux' in sys.platform), "Function can only run on Linux systems."
    print('Doing something.')

try:
    linux_interaction()
except AssertionError as error:
    print(error)
else:
    print('Executing the else clause.')
    (h, w) = frame.shape[:2]
try:
    with open('file.log') as file:
        read_data = file.read()
except:
    print('Could not open file.log')
    # Resizes the frame to suite the model requirements. Resizes the frame to 400X400 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (400, 400)), 0.007843, (400, 400), 127.5)

    network.setInput(blob)
    detections = network.forward()

    pos_dict = dict()
    coordinates = dict()
	validation_datagen = ImageDataGenerator(rescale=1./255)
	
	    # Focal length
	    F = 615

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2, j]

        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i])
		def linux_interaction():
		    # Define the function logic here
		    pass

            box = detection[0, 0, i, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Filtering only persons detected in the frame. Class Id of 'persons' is 15 which is vary every time
            if class_id == 15.00:

                # Draw bounding box for the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                label = "{}: {:.3f}%".format(labels[class_id], confidence * 100)
                print("{}".format(label))
# Function for nth fibonacci
# number - Space Optimisation
# Taking 1st two fibonacci numbers as 0 and 1
# Python3 code to demonstrate
# Characters which Occur in More than K Strings
# using set() + Counter() + loop + items()
from collections import Counter
from itertools import chain

def char_ex(strs, k):
	temp = (set(sub) for sub in strs)
	counts = Counter(chain.from_iterable(temp))
	return {chr for chr, count in counts.items() if count >= k}

# Initializing list
test_list = ['Gfg', 'ise', 'for', 'Geeks']

# printing original list
print("The original list is : " + str(test_list))

# Initializing K
K = 2

# Characters which Occur in More than K Strings
# using set() + Counter() + loop + items()
res = char_ex(test_list, K)
	
# printing result
print ("Filtered Characters are : " + str(res))

def fibonacci(n):
	a = 0
	b = 1
	
	# Check is n is less
	# than 0
	if n < 0:
		print("Incorrect input")
		
	# Check is n is equal
	# to 0
	elif n == 0:
		return 0
	
	# Check if n is equal to 1
	elif n == 1:
		return b
	else:
		for i in range(1, n):
			c = a + b
			a = b
			b = c
		return b

# Driver Program
print(fibonacci(9))

# This code is contributed by Saket Modi
# Then corrected and improved by Himanshu Kanojiya


                coordinates[i] = (startX, startY, endX, endY)
                coordinates[j] = (startX, startY, endX, endY)
                # Mid point of bounding box
                x_mid = round((startX+endX)/2,4)
                y_mid = round((startY+endY)/2,4)

                height = round(endY-startY,4)

                # Distance from camera based on triangle similarity
                distance = (165 * F)/height
                print("Distance(cm):{dist}\n".format(dist=distance))

                # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                
                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                pos_dict[i] = (x_mid_cm,y_mid_cm,distance)
    # Distance between every object detected in a frame
    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check if distance less than 2 metres or 200 centimetres not greter or less than that
                if dist < 305:
                    close_objects.add(i)
                    close_objects.add(j)

    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = np.array([0,0,255])
        else:
            COLOR = np.array([0,255,0])
        (startX, startY, endX, endY, 0) = coordinates[i]
try:
    linux_interaction()
except AssertionError as error:
    print(error)
else:
    try:
        with open('file.log') as file:
            read_data = file.read()
    except FileNotFoundError as fnf_error:
        print(fnf_error)
finally:
    print('Cleaning up, irrespective of any exceptions.')
        cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        # Convert cms to feet
        cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
def linux_interaction():
    assert ('linux' in sys.platform), "Function can only run on Linux systems."
    print('Doing something.')
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    # GCD of more than two (or array) numbers
# This function implements the Euclidian
# algorithm to find H.C.F. of two number

def find_gcd(x, y):
	while(y):
		x, y = y, x % y

	return x
	
	
l = [2, 4, 6, 8, 16]

num1=l[0]
num2=l[1]
gcd=find_gcd(num1,num2)

for i in range(2,len(l)):
	gcd=find_gcd(gcd,l[i])
	
print(gcd)

# Code contributed by Mohit Gupta_OMG

    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)
    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()


