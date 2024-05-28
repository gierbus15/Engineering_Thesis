import cv2
import numpy as np
import time
import os

class SheepDetector:
    def __init__(self, weights_path, config_path, names_path, video_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = self.load_classes(names_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.video_capture = cv2.VideoCapture(video_path)
        self.window_width = 1280
        self.window_height = 720
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video', self.window_width, self.window_height)
        self.sheep_data = {}
        self.feeders = {}
        self.eating_times = {}
        self.region_indices = {}  
        self.head_in_feeder = {}  

    def load_classes(self, file_path):
        with open(file_path, 'r') as f:
            classes = f.read().strip().split('\n')
        return classes

    def detect_sheep(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            boxes, confidences, class_ids = self.detect_objects(frame)
            self.draw_boxes(frame, boxes, confidences, class_ids)
            self.track_eating_time(boxes, class_ids)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.save_eating_times()
        self.video_capture.release()
        cv2.destroyAllWindows()

    def detect_objects(self, frame):
        boxes = []
        confidences = []
        class_ids = []
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return boxes, confidences, class_ids

    def draw_boxes(self, frame, boxes, confidences, class_ids):
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if label == 'pasnik':
                    feeder_id = self.get_feeder_id(x, y, w, h) 
                    self.feeders[feeder_id] = (x, y, w, h)
                    cv2.putText(frame, f'{feeder_id}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                elif label == 'owca_glowa':
                    head_id = self.get_head_id(x, y, w, h)
                    self.track_head_in_feeder(x, y, w, h)
                    cv2.putText(frame, f'{head_id}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def is_inside(self, box, point):
        x, y, w, h = box
        px, py = point
        return x <= px <= x + w and y <= py <= y + h

    def track_eating_time(self, boxes, class_ids):
        current_time = time.time() 
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(self.classes[class_ids[i]])
            if label == 'owca_glowa':
                for feeder_id, feeder_box in self.feeders.items():
                    if self.is_inside(feeder_box, (x, y)):
                        if feeder_id not in self.eating_times:
                            self.eating_times[feeder_id] = {}
                            self.create_time_file(feeder_id)
                        if 'start_time' not in self.eating_times[feeder_id]:
                            self.eating_times[feeder_id]['start_time'] = current_time
                            self.create_time_file(feeder_id)
                    else:
                        if feeder_id in self.eating_times:
                            start_time = self.eating_times[feeder_id]['start_time']
                            eating_duration = current_time - start_time
                            self.eating_times[feeder_id]['duration'] = eating_duration
                            print(f"Owca w {feeder_id} jadla przez {eating_duration} sekund.")

    def track_head_in_feeder(self, x, y, w, h):
        for feeder_id, feeder_box in self.feeders.items():
            if self.is_inside(feeder_box, (x, y)):
                if feeder_id not in self.head_in_feeder:
                    self.head_in_feeder[feeder_id] = time.time()
                    self.create_time_file(feeder_id)  
            else:
                if feeder_id in self.head_in_feeder and self.head_in_feeder[feeder_id] is not None:
                    start_time = self.head_in_feeder.pop(feeder_id)
                    end_time = time.time()
                    duration = end_time - start_time

    def save_eating_times(self):
        for feeder_id, data in self.eating_times.items():
            if 'duration' in data:
                duration = data['duration']
                with open(f'czas_{feeder_id}.txt', 'a') as f:
                    f.write(f"Owca w {feeder_id} jadla przez {duration:.2f} sekund.\n")

    def create_time_file(self, feeder_id):
        filename = f'czas_pasnik_{feeder_id}.txt'
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write("Czas jadania dla owiec w paśniku:\n")

    def get_feeder_id(self, x, y, w, h):
        for region_id, region_box in self.region_indices.items():
            if self.is_inside(region_box, (x, y)):
                return f'pasnik{region_id}'
        for region_id, region_box in self.region_indices.items():
            rx, ry, rw, rh = region_box
            if abs(rx - x) < 100 and abs(ry - y) < 100:  
                self.region_indices[region_id] = (x, y, w, h)  
                return f'pasnik{region_id}'
        region_id = len(self.region_indices) + 1
        if region_id <= 4:  
            self.region_indices[region_id] = (x, y, w, h)
            return f'pasnik{region_id}'
        else:
            None

    def get_head_id(self, x, y, w, h):
        for region_id, region_box in self.region_indices.items():
            if self.is_inside(region_box, (x, y)):
                return f'owca_glowa{region_id}'
        for region_id, region_box in self.region_indices.items():
            rx, ry, rw, rh = region_box
            if abs(rx - x) < 100 and abs(ry - y) < 100: 
                self.region_indices[region_id] = (x, y, w, h) 
                return f'owca_glowa{region_id}'
        region_id = len(self.region_indices) + 1
        if region_id <= 4:
            self.region_indices[region_id] = (x, y, w, h)
            return f'owca_glowa{region_id}'
        else:
            return f'owca_glowa{region_id-1}'

if __name__ == "__main__":
    weights_path = 'weights/yolov3_custom_last.weights'
    config_path = 'yolov3/yolov3_custom.cfg'
    names_path = 'yolov3/obj.names'
    video_path = 'C:/Users/Bartosz/Desktop/owca4/kam10_20231021150008_20231021160008.mp4'
    detector = SheepDetector(weights_path, config_path, names_path, video_path)
    detector.detect_sheep()
