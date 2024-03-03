This project allows you to train your own hand pose recognition model. It could also be technically be used for body pose recognition or facial expression recognition still using mediapipe, but I haven't experimented wuth that yet.

The project already has a example model loaded that can recognize rock, paper and scissors gestures. The example model was trained using only the right hand, but it can recognize the left hand as well although with a slightly worse precision. If you want to guarantee your model recognizes both hands well, make sure to include both hands (each at a time) while generating the images for the dataset.

In order to create your model follow these steps:

1. Run the "Collect_Images.py" script.
This script allows you to create your own dataset to train the model. When running the code, position your hand in front of the camera so that it is visible and press "q" to start recording. Move your hand around keeping the same gesture in order to create some variety on the dataset. Make sure only one hand is visible on the camera per frame or else the code will break. It might be wise to switch hands while keeping the same gesture in order to create more variety and so that the model can predict accurately for both hands.

Each frame will be saved as a separate image, the default setting is for 3 different hand gestures with 200 images each. You can change those parameters to your liking in the code.


2. Run the "Create_Dataset.py" script.
Here the images will be iterated and mediapipe will only collect the relevant information about the hands' landmarks. The landmarks represent the joints in your hand, and they are saved only as a coordinate on the screen.

The data of every landmark is saved into an array, and this is the dataset that we will be using to train the model (the position of the landmarks).

If the code is executed succesfully, there should appear a file in your directory called "data.pickle"


3. Run the "Train_Classifier.py" script.
Here the model will be created and trained using the "scikit-learn" library.

If the code is executed successfully there should appear a file named "model.p" in your directory.


4. Run the "Test_Classifier.py" script.
Here is where the magic happens. After training your model you can state the correct labels for each hand gesture that you collected in order using the "labels_dict" dictionary.

If everything is done right you should have your result :)
