import cv2
from DataLoader import load_data


def show_video(video_path="./dataset/video.mov", data_path='./dataset/test.txt'):
    cap = cv2.VideoCapture(video_path)
    raw = load_data(data_path)

    time_dict = {}
    for pData in raw:
        if int(pData.state[0]) is 1:
            continue
        if pData.frame in time_dict:
            time_dict[pData.frame].append(pData)
        else:
            time_dict[pData.frame] = [pData]

    count = 0
    while cap.isOpened():
        _, frame = cap.read()
        for person in time_dict[count]:
            cv2.circle(frame, tuple(int(i) for i in person.position), 2, (0, 0, 255), -1)
        cv2.imshow('window-name', frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    show_video()