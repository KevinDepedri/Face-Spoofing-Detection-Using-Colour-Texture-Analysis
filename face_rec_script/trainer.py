import cv2

"""
In this code we will generate a dataset that allows us to recognize one person. This will be helpful to test whether the 
anti-spoofing system works or not on that person.
"""

def generate_dataset(img, id, img_id):
    """
    This method generates multiple images of a person. They will be saved in the "dataset" directory
    :param img:
    :param id: integer assigned to a single person
    :param img_id: integer, each photo of a single person is marked by this id
    :return: None
    """

    cv2.imwrite("dataset/user."+str(id)+"."+str(img_id)+".jpg", img)

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors):
    """
    This method draws the boundaries (a box) around the detected face
    :param img: video capture
    :param classifier: faceCascade classifier
    :param scaleFactor:
    :param minNeighbors:
    :return: the coordinates (x,y) plus the width and the length of the rectangle
    """
    # First convert the image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the features of the image -> coordinates (x,y) plus width and height
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # draw the rectangle following the coordinates
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Face_detected", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def detect(img, faceCascade, img_id):
    """
    This method detects the features
    :param img: video capture
    :param faceCascade: classifier
    :param img_id: integer assigned to the photo
    :return: the detected face
    """
    # draw the boundaries around the detected face
    coords = draw_boundary(img, faceCascade, 1.1, 10)
    # If feature is detected, we have 4 coordinates which helps us draw the rectangle,
    # otherwise the length of coords will be 0
    if len(coords) == 4:
        # Creating the region of interest by using the coordinates x:x+w, y:y+h
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Create a unique ID to the user
        user_id = 1
        generate_dataset(roi_img, user_id, img_id)

    return img


# Load the haarcascade classifier -> only the frontal face one will be used
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the web-cam
video_capture = cv2.VideoCapture(0)

# Initialize img_id. This value increases as we obtain more images of the person
img_id = 0

# We want to obtain a total of 200 images of that person
while img_id != 201:
    # Read image from video
    _, img = video_capture.read()
    img = detect(img, faceCascade, img_id)
    cv2.imshow("Face", img)
    # increase the value of ID after one photo is saved into the dataset
    img_id += 1
    # If we want to interrupt the procedure before getting 200 images, we can press "q" on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release web-cam
video_capture.release()
# Destroy all windows
cv2.destroyAllWindows()