# Gesture Recognition Using a Neural Network

The code in the repository is for gesture recognition with the use of [Google Mediapipe Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer). It will recognize gestures from numbers 1-10. The approach will first get the hand landmarks of both hands and count the open fingers to get the gesture.

## How to Run

>- Add gestures to a folder named `fingers`
>- Run the `recognize.ipynb` file to get the output of the predicted gesture.

>Note:
>
>This approach was not utilized in our final implementation of the project since the approach had issues when the hands are slightly tilted. Therefore, in the final implementation we used the [Gesture recognition using a model](https://github.com/fusion-flow/gesture-recognizer-model) approach.
