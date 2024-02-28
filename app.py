from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import supervision as sv
import time
from pydantic import BaseModel
from typing import List
import numpy as np
class App:
    def __init__(self):
        self.detection_status = 0
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.cam = False
        self.cap = None
        self.model = None
        self.line_zone = None
        self.line_zone_annotator = None
        self.poly_zone = None
        self.poly_zone_annotate = None
        self.poly_zone_trigger = None
        self.prev_count = 0
        self.prev_start_x = None
        self.prev_start_y = None
        self.prev_end_x = None
        self.prev_end_y = None
        self.conf = 0.1
        self.iou = 0.4
        self.cam_num = 0
        self.listVel = [2]
        self.zones = None
        self.video_info = None
        self.polygon = None


    def initialize_stream(self):
        if self.cam_num == 0:
            url = "vid_kdn/3.mp4"
        elif self.cam_num == 1:
            url  ="vid_kdn/4.mp4"
        elif self.cam_num == 2:
            url = "vid_kdn/5.mp4"
        self.cap = cv2.VideoCapture(url)
        self.video_info =  sv.VideoInfo.from_video_path(url)

    def read_frame(self):
        if self.cap is None:
            self.initialize_stream()
        success, frame = self.cap.read()
        if not success or frame is None:
            print("Stream timeout or overread error")
            self.cap.release()
            self.cap = None
            return None, None
        return success, frame

    def load_model(self):
        try:
            if self.model is None:
                print("Attempting to load model...")
                self.model = YOLO("yolov8s.pt")
                print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model due to: {e}")
    
    def setup_Lineroi(self):
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None:
            start = sv.Point(self.start_x, self.start_y)
            end = sv.Point(self.end_x, self.end_y)

            self.line_zone = sv.LineZone(start=start, end=end)

            self.line_zone_annotator = sv.LineZoneAnnotator()

    def set_Lineroi(self, frame, detections,trackid):
        if self.start_x is not None and self.start_y is not None and self.end_x is not None and self.end_y is not None and self.line_zone is not None and self.line_zone_annotator is not None:
            if(trackid is not None):
                self.line_zone.trigger(detections=detections)
            self.line_zone_annotator.annotate(frame=frame, line_counter = self.line_zone)
            if(self.start_x != self.prev_start_x or self.start_y != self.prev_start_y or self.end_x != self.prev_end_x or self.end_y != self.prev_end_y):
                self.line_zone.out_count = self.prev_count
            self.prev_start_x = self.start_x
            self.prev_start_y = self.start_y
            self.prev_end_x = self.end_x
            self.prev_end_y = self.end_y
            self.prev_count= self.line_zone.out_count

        return frame
    
    def setup_Polygonroi(self):
        if self.polygon is not None:
            print("polygon", self.polygon)
            self.zones = sv.PolygonZone(polygon=self.polygon[0], frame_resolution_wh=self.video_info.resolution_wh)
            self.poly_zone_annotate = sv.PolygonZoneAnnotator(zone=self.zones, color=sv.Color.from_hex('#ff00ff'),  thickness=4)
    
    def set_Polygonroi(self, detections, frame):
        if self.zones is not None and self.poly_zone_annotate is not None:
            mask = self.zones.trigger(detections=detections)
            detections = detections[mask]
            frame = self.poly_zone_annotate.annotate(scene=frame)
        return frame
    
    def detection(self):
        frame_rate = 45.0  # Adjust this to match the video's frame rate
        frame_delay = 1.0 / frame_rate
        tracker = sv.ByteTrack()
        if self.detection_status == 1 and self.model is None:
            self.load_model()

        while True:
            
            frame_bytes = None
            success, frame = self.read_frame()
            height, width, _ = frame.shape
            self.size = (width, height)
            
            if not success or frame is None:
                continue
            if self.model is not None:
                results = self.model.track(frame,persist=True, conf=self.conf, show=False, verbose=False, iou=self.iou, classes=self.listVel)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[detections.area > 5000]
                detections = tracker.update_with_detections(detections)
                frame = self.annotate_frame(frame, detections)
                frame = self.set_Lineroi(frame, detections,results.boxes.id)
                frame = self.set_Polygonroi(detections,frame)
                frame_bytes = self.frame_to_bytes(frame)
            else:
                print("Model is not loaded, skipping frame processing.")
                self.load_model()
            if frame_bytes is not None:
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(frame_delay) 

    def annotate_frame(self, frame, detections):
        bounding_box_annotator = sv.BoundingBoxAnnotator(color=sv.ColorPalette.from_hex(['#f00000']), thickness=4)
        return bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)

    def frame_to_bytes(self, frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

app_state = App()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class VelModel(BaseModel):
    listVel: List[int]

class PointModel(BaseModel):
    x: int
    y: int
class PointsModel(BaseModel):
    points: List[PointModel]

@app.get("/detection")
def video_detection():
    return StreamingResponse(app_state.detection(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/start-detect")
def start_detect():
    app_state.detection_status = 1
    return {"message": "Detection started"}

@app.get("/swap-camera")
def swap_camera():
    app_state.cam = not app_state.cam
    app_state.cap = None
    return {"message": "Camera swapped"}

@app.get("/detectionStatus")
def get_detection_status():
    return {"detectionStatus": app_state.detection_status}

@app.get("/set-roi")
def set_Lineroi(start_x: int, start_y: int, end_x: int, end_y: int):
    app_state.start_x = start_x
    app_state.start_y = start_y
    app_state.end_x = end_x
    app_state.end_y = end_y
    app_state.setup_Lineroi()
    
    return {"message": "ROI set"}

@app.get("/set-conf")
def set_conf(conf: float):
    app_state.conf = conf
    return {"message": "Confidence set"}

@app.get("/set-iou")
def set_iou(iou: float):
    app_state.iou = iou
    return {"message": "IOU set"}


@app.post("/set-poly")
def set_poly(data: PointsModel):
    app_state.poly_zone_annotate = None
    app_state.polygon = [np.array([(point.x, point.y) for point in data.points])]
    app_state.setup_Polygonroi()
    return {"message":"s"}

@app.post("/set-vel")
def set_vel(vel: VelModel):
    app_state.listVel = vel.listVel
    return {"message":"s"}

@app.post("/get-vel")
def get_vel():
    return JSONResponse(content={"listVel": app_state.listVel})

@app.get("/get-conf")
def get_conf():
    return JSONResponse(content={"conf": app_state.conf})

@app.get("/get-iou")
def get_iou():
    return JSONResponse(content={"iou": app_state.iou})

@app.get("/next-cam")
def next_cam():
    app_state.cam_num += 1
    if app_state.cam_num == 3:
        app_state.cam_num = 0 
    app_state.cap = None
    return {"message": "Camera swapped"}

@app.get("/prev-cam")
def prev_cam():
    app_state.cam_num -= 1
    if app_state.cam_num == -1:
        app_state.cam_num = 3
    app_state.cap = None
    return {"message": "Camera swapped"}

