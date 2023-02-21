# <p align="center">Face Spoofing Detection Using Colour Texture Analysis</p> 

Most of the existing face recognition systems available are currently vulnerable to spoofing attacks. A spoofing attack occurs when someone tries to bypass a face recognition biometric system by presenting a fake face in front of the camera. The fake face can be obtained in different ways, from the most naive, like showing to the camera a picture or a video of the subject (using a printed sheet or another device screen), to the most elaborated, as building a 3D mask of the intrested subject. Face Spoofing Detection, also known as Presentation Attack Detection (PAD), aims at identifying when someone is trying to perform a Presentation Attack, avoiding this person to gain access to the desired data. PAD systems can rely on a wide group of possible approaches to counteract attackers and are generally more efficient when coupled together.

This repository is a student project developed by Kevin Depedri and Matteo Brugnera for the "Signal, Image and Video" course of the Master in Artificial Intelligent Systems at the University of Trento, a.y. 2021-2022.

The repository is composed of:
- A folder with the papers on which the implementation is based on
- A folder with our personal dataset used to train the SVM classifier
- Report-Presentation pdf
- Guide to build the needed virtual environment and to run the code (this readme)

The execution of the code is divided in phases as follows:
1. First, the acquisition device (webcam) is turned on, at this point the user has 30 seconds to perform a eye-blink (liveness clue)
    - If no eye-blink is detected, then the algorithm is terminated since nodoby is in front of the camera or, we are dealing with a static object (like a printed picture PA)
    - If a eye-blink is detected, a picture of the subject if front of the camera is acquired. 
2. After the eye-blink check, the acquired image is used to start the histogram-analysis phase of the algorithm
    - First, the input image is converted into both YCbCr and HSV color spaces, and each single channel is extracted
    - Then, for each channel extracted, the CoALBP and the LPQs descriptors are computed and concatenated (LPQ is computed each time under 3 different settings)
    - Finally, the descriptors of each channel (obtained in the previous step) are concatenated to build the final image descriptor
3. In the end, the final image descriptor is fed to a SVM-binary-classifier (trained on our personal dataset) that classifies the input image as Fake (Presentation Attack) or Real (Real User).

The SVM classifier has been trained on a restricted set of images acquired in static and normal indoor light condition. Due to the restricted amount of data present in the dataset we do not ensure a perfect generalization in all light conditions. Indeed, problems of detection (for eye-blink and face recognition) or wrong predictions (false positive or false negative) may arise when dealing with extreme light conditions (very bright or very dark). 

When testing the code please note that, while phase 1 (eye-blink) is evaluated depending on the sequence of frames acquired from a webcam, phase 2 and 3 can easily be tested on different inputs just changing the input path at ``line 63`` of the file ``main.py``. In this way, if the user is interested, it will be possible to text the histogram-analysis phase of the algorithm on specific input images. Exactly for this reason, we have attached two input images (one real and one fake) that you can use to test the algorithm on specific input. To do so you will need to change ``at line 63 of main.py`` the path ``'face_detected.jpg'`` in ``'face_detected_true.jpg'`` or ``'face_detected_fake.jpg'``.

****
# Building the virtual environment
To build the virtual environment follow the next few steps
  1. Clone the repository and get inside it
  ```shell
  git clone https://github.com/KevinDepedri/Face-Spoofing-Detection-Using-Colour-Texture-Analysis
  cd Face-Spoofing-Detection-Using-Colour-Texture-Analysis
  ```
  2. Launch a terminal instance from inside the folder
  3. Run the ``build_environment.sh`` file to start building the environment
  ```shell
  bash ./build_environment.sh
  ```
  
****
# Running the code
To run the code, once the virtual environment has been built, you will need to
  1. Activate the virtual environment
  ```shell
  ./venv/Scripts/activate
  ```
  2. Once the name of the environment is shown at the beginning of the current line in the terminal, then run the code
  ```shell
  python ./main.py <flags>
  ```
  When running the code different types of ``flags`` can be used to show the data computed by the algorithm. The list of available flags is:
  - ``-pe``, plot EAR (mean value) computed during the check of the eye-blink
  - ``-pc``, plot channels (RGB, Y, Cb, Cr, H, S, V) of the input image that will be used during the first phase of the histogram analysis
  - ``-pad``, plot all descriptors (channel: Y, Cb, Cr, H, S, V) computed during the histogram analysis
  - ``-pfd``, plot final descriptor (concatenation of channels: Y, Cb, Cr, H, S, V) computed at the end of the histogram analysis. The final decision of the SVM will be based on that final descriptor
  3. Wait for all the computations to be done. Please note that, each of the plots (enabled using the flags above) will temporarly stop the execution of the code. The execution will be resumed after that the plots have been closed. Finally, the program will return in the terminal if the image in input is Fake (attempt of Presentation Attack) or if it is a Real image.
