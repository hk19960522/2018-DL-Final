import cv2
from DataLoader import load_data


def show_result(video_path, sample_frame, input_traces, target_traces, prediction):
    def draw_coords(ss, color):
        for pp in ss:
            p = (int(pp[0][0] * 1920), int(pp[0][1] * 1080))
            for i in range(1, len(pp)):
                p1 = (int(pp[i][0] * 1920), int(pp[i][1] * 1080))
                cv2.line(frame, p, p1, color, 5)
                p = p1

    time_dict = {}
    for t, inp, tar, pre in zip(sample_frame, input_traces, target_traces, prediction):
        inp_data = inp[:, :, :2].data.numpy()
        tar_data = tar[:, :, :2].data.numpy()
        pre_data = pre[:, :, :2].data.numpy()
        time_dict[t] = (inp_data, tar_data, pre_data)

    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        _, frame = cap.read()
        if count in time_dict:
            # print(len(time_dict[count]))
            draw_coords(time_dict[count][0], (255, 0, 0))  # blue
            draw_coords(time_dict[count][1], (0, 255, 0))  # green
            draw_coords(time_dict[count][2], (0, 0, 255))  # red

        cv2.imshow('show_result', frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


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