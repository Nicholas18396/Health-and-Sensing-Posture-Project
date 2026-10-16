# Ref: https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/

import cv2
import time
import math as m
import mediapipe as mp
import argparse
import matplotlib.pyplot as plt
import winsound
import threading
import os

sound_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Fix_Your_Posture.wav"
)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def findDistance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Distance between the two points.
    """
    dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    """
    Calculate the angle between two points with respect to the y-axis.

    Args:
        x1, y1: Coordinates of the first point.
        x2, y2: Coordinates of the second point.

    Returns:
        Angle in degrees.
    """

    return -1*m.degrees(m.atan2(x2-x1, y1-y2))

    '''theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi) * theta
    return degree'''

def sendWarning(x=None):
    """
    Placeholder function for sending a warning.
    """
    print("Posture Violation! Sit Straight Up!!")

    winsound.PlaySound(
        sound_file,
        winsound.SND_FILENAME
    )


    pass

def list_available_cameras(max_cameras=10):
    """
    Probe camera indices 0..max_cameras-1 and return the ones that
    actually open and deliver a frame.

    Args:
        max_cameras: Highest index (exclusive) to probe.

    Returns:
        List of valid camera indices, e.g. [0, 1].
    """
    available = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(index)
        cap.release()
    return available

def select_camera():
    """
    Scan for available cameras and let the user pick one interactively.

    Returns:
        The chosen camera index (int).
    """
    print("Scanning for available cameras...")
    cameras = list_available_cameras()

    if not cameras:
        print("No cameras found. Defaulting to camera 0.")
        return 0

    if len(cameras) == 1:
        print(f"Only one camera found (index {cameras[0]}). Using it.")
        return cameras[0]

    print("Available cameras:")
    for idx in cameras:
        print(f"  [{idx}] Camera {idx}")

    while True:
        choice = input(f"Select a camera index {cameras}: ").strip()
        try:
            choice_int = int(choice)
            if choice_int in cameras:
                return choice_int
            print("That index isn't in the list, try again.")
        except ValueError:
            print("Please enter a number.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Posture Monitor with MediaPipe')
    parser.add_argument('--video', type=str, default=None, help='Path to the input video file. If not provided, the webcam will be used.')
    parser.add_argument('--offset-threshold', type=int, default=100, help='Threshold value for shoulder alignment.')
    parser.add_argument('--neck-angle-threshold', type=int, default=25, help='Threshold value for neck inclination angle.')
    parser.add_argument('--torso-angle-threshold', type=int, default=10, help='Threshold value for torso inclination angle.')
    parser.add_argument('--time-threshold', type=int, default=3, help='Time threshold for triggering a posture alert.')
    return parser.parse_args()

def main(video_path="test.mp4", offset_threshold=100, neck_angle_threshold=25, torso_angle_threshold=10, time_threshold=3):
    # Initialize frame counters.
    good_frames = 0
    bad_frames = 0
    good_frames_int = 0
    bad_frames_int = 0

    # Font type.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Colors.
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    white = (255, 255, 255)

    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    if isinstance(video_path, int):
        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        source_desc = f"camera index {video_path}"
    else:
        cap = cv2.VideoCapture(video_path)
        source_desc = f"video file '{video_path}'"

    if not cap.isOpened():
        print(f"Could not open {source_desc}. Check the path/connection and try again.")
        return

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    time_data = []
    neck_data = []
    torso_data = []
    start_time = time.time()
    last_warning_time = 0
    good_time = 0.0
    bad_time = 0.0
    bad_streak = 0.0
    prev_time = time.time()
    is_camera = isinstance(video_path, int)

    while True:
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 30  # fall back if the camera doesn't report a valid fps

        now = time.time()
        dt = (now - prev_time) if is_camera else (1/fps)
        prev_time = now

        # Get height and width of the frame.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        if lm is None:
            cv2.putText(image, "No person detected", (10, 30), font, 0.9, red, 2)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        # Right shoulder.
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < offset_threshold:
            cv2.putText(image, str(int(offset)) + ' Shoulders aligned', (w - 280, 30), font, 0.6, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Shoulders not aligned', (w - 280, 30), font, 0.6, red, 2)

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Update Plot Data
        elapsed_time = time.time() - start_time
        time_data.append(elapsed_time)
        neck_data.append(neck_inclination)
        torso_data.append(torso_inclination)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, white, 2)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, white, 2)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, white, 2)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string_neck = 'Neck inclination: ' + str(int(neck_inclination))
        angle_text_string_torso = 'Torso inclination: ' + str(int(torso_inclination))

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        if abs(neck_inclination) < neck_angle_threshold:
            #and torso_inclination < torso_angle_threshold:

            bad_streak = 0.0
            good_time += dt

            cv2.putText(image, angle_text_string_neck, (10, 30), font, 0.6, light_green, 2)
            cv2.putText(image, angle_text_string_torso, (10, 60), font, 0.6, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 2)

        else:
            bad_time += dt
            bad_streak += dt

            if bad_streak >= time_threshold:
                current_time = time.time()

                if current_time - last_warning_time >= 5:
                    threading.Thread(target=sendWarning, daemon=True).start()
                    last_warning_time = current_time

            cv2.putText(image, angle_text_string_neck, (10, 30), font, 0.6, red, 2)
            cv2.putText(image, angle_text_string_torso, (10, 60), font, 0.6, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

            # Join landmarks.
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 2)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 2)

        # Pose time.
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 50), font, 0.9, green, 2)
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', image)

        # Exit the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows.
    cap.release()
    cv2.destroyAllWindows()

    # Create Plots
    plt.figure(figsize = (12,6))

    neck_color = 'tab:blue'
    torso_color = 'tab:orange'
    
    plt.plot(time_data, neck_data, color = neck_color, label = 'Neck Inclination')
    plt.plot(time_data, torso_data, color = torso_color, label = 'Torso Inclination')

    plt.axhline(y = neck_angle_threshold, color = neck_color, linestyle = '--', label = 'Neck Threshold')
    plt.axhline(y = torso_angle_threshold, color = torso_color, linestyle = '--', label = 'Torso Threshold')

    plt.xlabel('Time (s)')
    plt.ylabel('Inclination Angle (degrees)')
    plt.title('Posture Inclination Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    args = parse_arguments()

    if args.video is None:
        video_source = select_camera()
    elif args.video.isdigit():
        video_source = int(args.video)
    else:
        video_source = args.video
    
    print("Arguments:")
    print(f"Video: {args.video}")
    print(f"Offset Threshold: {args.offset_threshold}")
    print(f"Neck Angle Threshold: {args.neck_angle_threshold}")
    print(f"Torso Angle Threshold: {args.torso_angle_threshold}")
    print(f"Time Threshold: {args.time_threshold}")

    main(
        video_path=video_source,
        offset_threshold=args.offset_threshold,
        neck_angle_threshold=args.neck_angle_threshold,
        torso_angle_threshold=args.torso_angle_threshold,
        time_threshold=args.time_threshold,
    )
