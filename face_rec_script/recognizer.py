import cv2

"""
In this code we are able to recognize the person if he/she stands in front of the web-cam
"""


def draw_boundary(img, classifier, scaleFactor, minNeighbors, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # Predict the id of the user
        id, _ = clf.predict(gray_img[y:y+h, x:x+w])
        # If the ID corresponds to the trained person then draw a rectangle around his/her face
        if id == 1:
            cv2.putText(img, "Bru", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords

# Method to recognize the person
def recognize(img, clf, faceCascade):
    """
    This method makes sure that the system recognizes the person
    :param img: video capture
    :param clf: classifier
    :param faceCascade: classifier
    :return: recognized person if in front of the web-cam
    """
    draw_boundary(img, faceCascade, 1.1, 10, clf)
    return img


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)

while True:
    _, img = video_capture.read()
    img = recognize(img, clf, faceCascade)

    cv2.imshow("face detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()