from ultralytics import YOLO

import supervision as sv
import pandas as pd
import numpy as np

import cv2
import torch
import datetime

from tqdm import tqdm


print('CUDA available ->', torch.cuda.is_available())


def process_frame(frame: np.ndarray, frame_number: int, data_trace: pd.DataFrame, data_line: pd.DataFrame) -> np.ndarray:
        # Прохождение одного фрейма через модель и вычленение детекций
        results = model(
             frame,
             verbose=False,
             conf=0.3,
             iou=0.3
        )
        results = results[0]    # [predictions[boxes]]
        
        detections = sv.Detections.from_ultralytics(results)            
        detections = tracker.update_with_detections(detections)

        for i in range(len(detections.xyxy)):
            data_trace = pd.concat([
                  data_trace,
                  pd.DataFrame({
                        'x_min':      [float(detections.xyxy[i][0])],           # наименьшая координата бокса по x
                        'y_min':      [float(detections.xyxy[i][1])],           # наименьшая координата бокса по y
                        'x_max_':     [float(detections.xyxy[i][2])],           # наибольшая координата бокса по x
                        'y_max':      [float(detections.xyxy[i][3])],           # наибольшая координата бокса по y
                        'x_center':   [(float(detections.xyxy[i][0]) +     
                                       float(detections.xyxy[i][2])) / 2],      # координата центра боксов по оси x
                        'y_center':   [(float(detections.xyxy[i][1]) + 
                                       float(detections.xyxy[i][3])) / 2],      # координата центра боксов по оси y
                        'class_id':   [int(detections.class_id[i])],            # значение id класса среди всех
                        'confidence': [float(detections.confidence[i])],        # уверенность модели в предсказании
                        'tracker_id': [int(detections.tracker_id[i])],          # значение id бъекта (трекер)
                        'class_name': [str(detections.data['class_name'][i])],  # наименование класса
                        'current_frame': [int(frame_number)],                   # номер кадра детекции объекта
                        'total_frames': [int(video_info.total_frames)],         # сколько всего кадров
                        'file_name': [str(SOURCE_VIDEO_PATH)],
                        'process_dttm': [datetime.datetime.now()],              # время записи информации по объекту
                  })
            ])
     
        label = [f"{tracker_id} {dtype['class_name']} {confidence:0.2f}"
                 for xyxy, mask, confidence, class_id, tracker_id, dtype
                 in detections]

        frame = trace_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, label)
        frame = box_annotator.annotate(frame, detections=detections)

        crossed_in, crossed_out = line_zone.trigger(detections)
        line_zone_annotator.annotate(frame, line_counter=line_zone)

        data_line = pd.concat([
              data_line,
              pd.DataFrame({
                    'in_count': [int(np.sum(crossed_in))],            # прошедшие inside
                    'out_count': [int(np.sum(crossed_out))],          # прошедшие outside
                    'frame_number': [int(frame_number)],              # номер кадра прохождения объектов
                    'total_frames': [int(video_info.total_frames)],   # сколько всего кадров
                    'process_dttm': [datetime.datetime.now()]         # время записи информации по объекту
              })
        ])

        return (frame, data_trace, data_line)


SOURCE_VIDEO_PATH = "08506_1_Краснопресненская_наб._д.18к1_в_центр_Краснопресненская_наб._д.18к1_в_цен_12.09.2024_09.30.00-12.09.2024_09.45.00.mp4"
TARGET_VIDEO_PATH = "result.mp4"

box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=1)
trace_annotator = sv.TraceAnnotator(thickness=2)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

START = sv.Point(31, 400)
END = sv.Point(1500, 500)

line_zone = sv.LineZone(start=START, end=END, triggering_anchors=(sv.Position.CENTER,))

tracker = sv.ByteTrack()

model = YOLO("best.pt")

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEO_PATH
)

window_name = 'Object tracing'

data_trace = pd.DataFrame(
      columns=[
            'x_min',
            'y_min',
            'x_max',
            'y_max',
            'x_center',
            'y_center',
            'class_id',
            'confidence',
            'tracker_id',
            'class_name',
            'current_frame',
            'total_frames',
            'file_name',
            'process_dttm'
            ]
)

data_line = pd.DataFrame(
      columns=[
            'in_count',
            'out_count',
            'frame_number',
            'total_frames',
            'process_dttm'
            ]
)

for i, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
    annotated_frame, data_trace, data_line = process_frame(frame, i, data_trace, data_line)
    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

data_trace.to_csv('detections.csv', index=False)
data_line.to_csv('line_count.csv', index=False)