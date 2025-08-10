import cv2
import mediapipe as mp


class HandDetector:
    """
    Detects and annotates hand landmarks in video frames using MediaPipe.

    Parameters:
        model_complexity (int): Complexity of the hand landmark model (0 or 1).
        min_detection_confidence (float): Minimum confidence value for hand detection.
        min_tracking_confidence (float): Minimum confidence value for hand tracking.

    Usage:
        detector = HandDetector()
        annotated_frame, results = detector.process_frame(frame)

    The class customizes the thumb landmark and connection colors for better visibility.
    """
    def __init__(self, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Mediapipe helpers
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # Initialize Mediapipe Hands
        self.hands = self.mp_hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Create custom styles for thumb landmarks/connections
        self.default_landmark_style = self.mp_drawing_styles.get_default_hand_landmarks_style()
        self.default_connection_style = self.mp_drawing_styles.get_default_hand_connections_style()

        self._customize_thumb_color()

    def _customize_thumb_color(self):
        """Only used for changing the color of the THUMB landmarks and connections as the default color is too bright"""
        
        custom_landmark_style = self.mp_drawing.DrawingSpec(color=(147, 20, 255), thickness=4, circle_radius=3) # Pink style for landmarks
        for idx in range(2, 5): # Thumb indices written in mediapipe documentation
            self.default_landmark_style[idx] = custom_landmark_style

        custom_connection_style = self.mp_drawing.DrawingSpec(color=(147, 20, 255), thickness=2, circle_radius=2) # Pink style for thumb connections
        thumb_connections = [(1, 2), (2, 3), (3, 4)] # Connections for thumb
        for conn in thumb_connections:
            self.default_connection_style[conn] = custom_connection_style

    def process_frame(self, frame):
        """Process the frame and return results + annotated frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)

        annotated_frame = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks: # Iterate through each detected hand
                # Draw landmarks and connections on the annotated frame
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.default_landmark_style, # Custom style for landmarks (changed thumb)
                    connection_drawing_spec=self.default_connection_style # Custom style for connections
                )

        return annotated_frame, results


class FingerCounter:
    def __init__(self):
        # Landmark indices for fingertips in Mediapipe
        self.finger_tips = [4, 8, 12, 16, 20]

    def count_fingers(self, hand_landmarks, handedness):
        """Count fingers for one hand"""
        fingers = []

        # This if else statement is for the thumb only
        # We branch because the thumb sticks out sideways. Which direction “out” is depends on whether the detected hand is left or right
        # ex: self.finger_tips[0] is 4 (thumb tip), self.finger_tips[0] - 1 is 3 (the landmark just before the tip)
        # Compares the x coordinate of the thumb tip (landmark[4].x) with the x of its preceding joint (landmark[3].x)
        # The logic: for a right hand facing the camera with fingers up, the thumb, when extended, 
        # will appear to the left of its inner joint (smaller x). So tip.x < joint.x --> thumb extended --> append 1. Otherwise 0.
        # The inverted logic applies for a left hand
        if handedness == "Right":
            fingers.append(1 if hand_landmarks.landmark[self.finger_tips[0]].x < hand_landmarks.landmark[self.finger_tips[0] - 1].x else 0)
        else:  # Left hand
            fingers.append(1 if hand_landmarks.landmark[self.finger_tips[0]].x > hand_landmarks.landmark[self.finger_tips[0] - 1].x else 0)

        # This is for the other fingers
        # The logic: Loop over the other fingertips 8, 12, 16, 20. 
        # Compares y of the fingertip to the y of the landmark two indices before (tip - 2)
        # Example: for index finger, tip=8 and tip-2=6
        # .y is normalized vertical position (0 = top, 1 = bottom). So smaller y means higher (closer to top of image).
        # if a finger is extended (pointing up), the fingertip will be above (smaller y) than the tip - 2. 
        # So tip.y < tip-2.y means that finger is raised --> 1. If the finger is folded down toward the palm, the tip will be below or near tip -2 --> 0.
        for tip in self.finger_tips[1:]:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

        return sum(fingers)


class VideoCaptureHandler:
    """
    Handles webcam video capture, processes each frame for hand detection and annotation,
    and displays the results in a window.

    Parameters:
        detector (HandDetector): Instance for detecting and annotating hands.
        counter (FingerCounter): Instance for counting raised fingers.
        camera_index (int): Index of the webcam to use (default is 0).

    Usage:
        video_handler = VideoCaptureHandler(detector, counter)
        video_handler.run()

    The run() method reads frames from the webcam, flips them for selfie view,
    detects hands, counts fingers, overlays results, and displays the annotated frames.
    Press ESC to exit.
    """
    def __init__(self, detector, counter, camera_index=0):
        self.detector = detector
        self.counter = counter
        self.cap = cv2.VideoCapture(camera_index)

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return

        while True:
            success, frame = self.cap.read()
            if not success:
                print("Ignoring empty camera frame")
                continue

            # Flip BEFORE detection, this is to avoid the text and hands being opposite
            # If we didn't flip before, when we raise our right hand, the text would say "Left Hand" and vice versa
            frame = cv2.flip(frame, 1)

            annotated_frame, results = self.detector.process_frame(frame)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    count = self.counter.count_fingers(hand_landmarks, handedness)

                    cv2.putText(annotated_frame, f"{handedness} Hand: {count}",
                                (10, 40 + idx * 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2)

            cv2.imshow('MediaPipe Hands', annotated_frame)

            if cv2.waitKey(5) & 0xFF == 27: # If ESC is pressed (27 is the ASCII code for the ESC key), break the loop. waitKey(5) = checks every 5 milliseconds for ESC key press
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Short main function to run the video capture and hand detection
def main():
    detector = HandDetector()
    counter = FingerCounter()
    video_handler = VideoCaptureHandler(detector, counter)
    video_handler.run()

if __name__ == "__main__":
    main()