import os
from time import sleep

import cv2
import torch


def draw_coords(frame, ss, color, start=None):
    for it, pp in enumerate(ss):
        if start is None:
            p = (int(pp[0][0] * 1920), int(pp[0][1] * 1080))
        else:
            p = int(start[it, 0] * 1920), int(start[it, 1] * 1080)
            print(p)
        for i in range(1, len(pp)):
            p1 = (int(pp[i][0] * 1920), int(pp[i][1] * 1080))
            cv2.line(frame, p, p1, color, 5)
            p = p1
    return ss[:, -1, :]


def show_result(video_path, sample_frame, input_traces, target_traces, prediction):
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
        if count in time_dict:  # ped_num, frame, pos
            # print(len(time_dict[count]))
            start = draw_coords(frame, time_dict[count][0], (255, 0, 0))  # blue
            draw_coords(frame, time_dict[count][1], (0, 255, 0), start)  # green
            draw_coords(frame, time_dict[count][2], (0, 0, 255), start)  # red

        cv2.imshow('show_result', frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_CUHK(frame_directory, train_traces, target_traces, prediction, sample_frame):
    time_dict = {}
    for t, inp, tar, pre in zip(sample_frame, train_traces, target_traces, prediction):
        inp_data = inp[:, :, :2].data.numpy()
        tar_data = tar[:, :, :2].data.numpy()
        pre_data = pre[:, :, :2].data.numpy()
        time_dict[t] = (inp_data, tar_data, pre_data)

    for frame_count in range(0, max(time_dict)+1, 20):
        path = os.path.join(frame_directory, '{:06d}.jpg'.format(int(frame_count)))
        frame = cv2.imread(path)

        if frame_count in time_dict:
            print(time_dict[frame_count][0].shape)
            start = draw_coords(frame, time_dict[frame_count][0], (255, 0, 0))  # blue
            draw_coords(frame, time_dict[frame_count][1], (0, 255, 0), start)  # green
            draw_coords(frame, time_dict[frame_count][2], (0, 0, 255), start)  # red
        else:
            continue

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, str(frame_count), (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('show_result', frame)
        sleep(0.5)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    d1, d2, d3 = torch.load('./dataset/AnnotationDataset.pkl')
    print(d1.size(), d2.size(), len(d3))
    show_CUHK('./dataset/Frame/', d1, d2, d2, d3)