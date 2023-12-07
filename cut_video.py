import cv2

def save_frames_as_images(video_path, output_folder,frame_interval):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        frame_count += 1
        if frame_count % frame_interval == 0:
            image_name = f"{output_folder}/{frame_count//frame_interval:04d}.jpg"
            cv2.imwrite(image_name, frame)

    video_capture.release()
    cv2.destroyAllWindows()

# 동영상 파일 경로 설정
video_path = 'C:/Users/sunp/Desktop/nlp/라바_시즌1_16화_스파게티.mp4'  # 동영상 파일 경로
output_folder = 'C:/Users/sunp/Desktop/nlp/larva16'  # 저장할 폴더
frame_interval = 3  # 프레임 간격 설정

# 프레임을 이미지로 저장
save_frames_as_images(video_path, output_folder, frame_interval)
