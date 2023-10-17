import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

def calculate_face_direction(face_landmarks, image_width, image_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),           # 鼻の先端
        (0.0, -330.0, -65.0),       # 顎
        (-225.0, 170.0, -135.0),    # 左目の内側の点
        (225.0, 170.0, -135.0),     # 右目の内側の点
        (-150.0, -150.0, -125.0),   # 左耳の先端
        (150.0, -150.0, -125.0)     # 右耳の先端
    ], dtype=np.float64)

    image_points = []
    for landmark in face_landmarks:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        image_points.append((x, y))
    
    image_points = np.array(image_points, dtype=np.float64)

    # カメラ行列を設定
    focal_length = image_width
    center = (image_width // 2, image_height // 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # 畸 distortion coefficients = 0 に設定

    # solvePnP関数で姿勢を推定
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 回転ベクトルをユークリッド角度に変換
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles, _ = cv2.decomposeProjectionMatrix(rotation_matrix)[:3]

    return euler_angles


def main():
    st.title("MediaPipeを使用した顔の向き推定アプリ")
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
                    euler_angles = calculate_face_direction(face_landmarks.landmark, frame.shape[1], frame.shape[0])
                    cv2.putText(frame, f"Roll: {euler_angles[0][0]:.2f}, Pitch: {euler_angles[1][0]:.2f}, Yaw: {euler_angles[2][0]:.2f}",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame)

        video_capture.release()
    else:
        st.write("カメラは停止しています。")

if __name__ == "__main__":
    main()
