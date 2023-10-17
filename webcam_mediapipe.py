import streamlit as st
import cv2
import mediapipe as mp

def main():
    st.title("MediaPipeを使用した顔のランドマーク表示アプリ")
    run = st.checkbox("カメラを開始する")

    FRAME_WINDOW = st.image([])

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.2, min_tracking_confidence=0.2)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    if run:
        video_capture = cv2.VideoCapture(3)

        while run:
            _, frame = video_capture.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec)

            FRAME_WINDOW.image(frame)

        video_capture.release()
    else:
        st.write("カメラは停止しています。")

if __name__ == "__main__":
    main()
