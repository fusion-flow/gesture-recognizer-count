import mediapipe as mp
import cv2

#Initializations: static code
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils



class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        #when the mediapipe is first started, it detects the hands. After that it tries to track the hands
        #as detecting is more time consuming than tracking. If the tracking confidence goes down than the
        #specified value then again it switches back to detection
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)


    def findHandLandMarks(self, image, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # mediapipe needs RGB
        results = self.hands.process(image)
        landMarkList = []
        num_hands = 1

        if results.multi_handedness:
            num_hands = len(results.multi_handedness)
            
            for i in range(num_hands):
                label = results.multi_handedness[i].classification[0].label
                #account for inversion in webcams
                if label == "Left":
                    label = "Right"
                elif label == "Right":
                    label = "Left"


        if results.multi_hand_landmarks:  # returns None if hand is not found
            for i in range(num_hands):
                landmark_hand = []
                hand = results.multi_hand_landmarks[i] #results.multi_hand_landmarks returns landMarks for all the hands
                print(i, hand)
                for id, landMark in enumerate(hand.landmark):
                    # landMark holds x,y,z ratios of single landmark
                    h, w, c = originalImage.shape  # height, width, channel for image
                    xPos, yPos = int(landMark.x * w), int(landMark.y * h)
                    landmark_hand.append([id, xPos, yPos, label])

                if draw:
                    mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)
            
                landMarkList.append(landmark_hand)

        return landMarkList