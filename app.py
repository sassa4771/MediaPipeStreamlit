import streamlit as st
import mediapipe as mp
import cv2

# MediaPipeの初期化
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Streamlitアプリの設定
st.title("顔のランドマーク表示アプリ")
st.sidebar.header("設定")
confidence_threshold = st.sidebar.slider("検出信頼度の閾値", 0.0, 1.0, 0.2)

# Webカメラの起動
cap = cv2.VideoCapture(3)

while cap.isOpened():
    ret, frame = cap.read()

    # 画像をMediaPipeのFace Detectionモデルに渡して顔を検出
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 検出された顔にランドマークを描画
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            if detection.score[0] > confidence_threshold:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 検出された顔の部分画像を切り取る
                face_image = frame[y:y+h, x:x+w]

                # 切り取った顔の画像をStreamlitのウィンドウに表示
                st.image(face_image, channels="BGR", use_column_width=True, caption="Detected Face")


# カメラを解放
cap.release()
