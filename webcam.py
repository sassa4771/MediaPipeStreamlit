import streamlit as st
import cv2

def main():
    st.title("Webカメラを表示するアプリ")
    run = st.checkbox("カメラを開始する")

    FRAME_WINDOW = st.image([])

    if run:
        video_capture = cv2.VideoCapture(3)

        while run:
            _, frame = video_capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        video_capture.release()
    else:
        st.write("カメラは停止しています。")

if __name__ == "__main__":
    main()
