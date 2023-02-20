import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import full_histogram
import create_plot
import eye_blink
from time import sleep
from argparse import ArgumentParser

parser = ArgumentParser(prog='Face Spoofing Detection Using Colour Texture Analysis')
parser.add_argument('-pe', '--plot_ear_mean', action='store_true', default=False,
                        help='Plot the mean EAR obtained during the eye-blinking check')
parser.add_argument('-pc', '--plot_channels', action='store_true', default=False,
                        help='Plot all the different channels of the input image that will be analyzed')
parser.add_argument('-pad', '--plot_all_descriptors', action='store_true', default=False,
                        help='Plot all the the descriptors obtained during the histogram-analysis')
parser.add_argument('-pfd', '--plot_final_descriptor', action='store_true', default=False,
                        help='Plot the final descriptor obtained at the end of the histogram-analysis')
args = parser.parse_args()
print('Input arguments:', end=' ')
print(args)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# check the eye blinking
possible_blinking = eye_blink.eye_blinking(cap)
print("Blinking check terminated")
EAR_mean = create_plot.get_graph(args.plot_ear_mean)
print('The mean EAR value is: {}'.format(EAR_mean))

if possible_blinking:
    print("Blink successfully detected! Starting histogram analysis...")
    print("Acquisition of a picture for histogram analysis")
    sleep(0.5)

    # save the face in order to obtain the full histogram and checking whether the face is fake or not
    face_not_detected = True
    while face_not_detected:
        _, img = cap.read()
        cv2.imshow("Face", img)
        key = cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
        for (x, y, w, h) in faces:
            roi_color = img[y:y + h, x:x + w]
            print("Face found -> Saving it as 'face_detected.jpg'")
            cv2.imwrite('face_detected.jpg', roi_color)
            face_not_detected = False

    image = cv2.imread('face_detected.jpg')
    cap.release()
    cv2.destroyAllWindows()

    # loading the already trained model in order to make a prediction of the face we found
    pick = open('model.sav', 'rb')
    model = pickle.load(pick)
    pick.close()

    # compute the histogram of the face we found and saved. Change the input here below with 'face_detected_fake' or
    # 'face_detected_true' if you want to try this part of the algorith with the images provided in the repository
    face = full_histogram.final_function('face_detected.jpg', args.plot_channels, args.plot_all_descriptors,
                                         args.plot_final_descriptor)
    face = face.reshape(1, -1)  # we have to reshape it since we are working only with one sample
    prediction = model.predict(face)

    # print the result we get
    categories = ['True', 'Fake']  # we have two possible solutions
    print('The face detected is: {}'.format(categories[prediction[0]]))
else:
    print("No face/blinking detected within 30sec! Closing the app...")
    cap.release()
    cv2.destroyAllWindows()



