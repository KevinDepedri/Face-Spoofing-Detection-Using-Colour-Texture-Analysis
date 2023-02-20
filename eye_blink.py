from scipy.spatial import distance as dist
import cv2
import dlib
import time
import csv


def get_blinking_ear(eye_points, facial_landmarks, frame):
    """
    This method calculates the EAR of every frame of the video
    :param eye_points: coordinates of the eyes
    :param facial_landmarks: landmarks placed around the eyes
    :param frame: image we want to analyze
    :return: the EAR of the image
    """
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    hor_distance = dist.euclidean(left_point, right_point)

    top_left = (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y)
    bottom_left = (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
    ver_distance_left = dist.euclidean(top_left, bottom_left)

    top_right = (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y)
    bottom_right = (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y)
    ver_distance_right = dist.euclidean(top_right, bottom_right)

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line_left = cv2.line(frame, top_left, bottom_left, (0, 255, 0), 2)
    ver_line_right = cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)

    ear = (ver_distance_left + ver_distance_right) / (2 * hor_distance)
    return ear


def eye_blinking(cap):
    """
    Detects whether an eye blink has occurred
    :param cap: video capture
    :return: Boolean that tells if a blink has occurred
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_SIMPLEX

    eye_open = False
    eye_closed = False

    #time limit of 30 seconds
    start_time = int(time.time())
    finish_time = int(time.time()) + 30

    iteration = 0
    header = ["EAR", "ITERATION"]

    #we save each and every EAR value for each iteration
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=header)
        csv_writer.writeheader()

    eye_threshold = 0.2

    #either the 30 seconds pass or both eye_open and eye_closed have been detected
    while (start_time != finish_time) and not(eye_open and eye_closed):
        start_time = int(time.time())
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        cv2.putText(frame, f"{round(finish_time-start_time, 1)}", (-5, 50), font, 2, (0, 255, 255))

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye_ratio = get_blinking_ear([36, 37, 38, 39, 40, 41], landmarks, frame)
            right_eye_ratio = get_blinking_ear([42, 43, 44, 45, 46, 47], landmarks, frame)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if blinking_ratio > eye_threshold:
                eye_open = True
            else:
                cv2.putText(frame, "BLINKING", (80, 50), font, 2, (0, 0, 255))
                eye_closed = True

            with open('data.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=header)

                info = {
                    "EAR": blinking_ratio,
                    "ITERATION": iteration
                }

                csv_writer.writerow(info)
                iteration += 1

        cv2.imshow("Face", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # cap.release()
    # cv2.destroyAllWindows()
    return eye_open and eye_closed # it's true if the eyes blinked once, otherwise false

