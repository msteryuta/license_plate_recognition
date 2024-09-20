# 匯入所需的函式庫
import cv2
from ultralytics import YOLO
import re
import math
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image

# 計算中心點座標
def calculate_center(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# 計算兩點之間的距離
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 確認傳入的車牌框是否在車輛框內
def get_car(license_plate, vehicles):
    x1, y1, x2, y2, score = license_plate
    for idx, vehicle in enumerate(vehicles):
        car_x1, car_y1, car_x2, car_y2 = vehicle[0]
        if car_x1 < x1 and car_y1 < y1 and car_x2 > x2 and car_y2 > y2:
            return (car_x1, car_y1, car_x2, car_y2)

    return None

# 將車牌影像轉換成文字
def image_to_text(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return process_return(generated_text)

# 檢查轉換後的字串是否符合車牌編碼規則
def process_return(text):
    try:
        text = text.upper()
        text = text.replace('I', '1').replace('O', '0').replace(' ', '').replace('4', 'A')  # 台灣車牌沒有I、O(英文字母)及4，因此將其更換成易於辨識的字元
        text = re.sub(r'[^A-Z0-9]', '', text)
        patterns = [
            re.compile(r'^[A-Z]{3}\d{4}$'),        # 英英英-數數數數
            re.compile(r'^[A-Z]{2}\d{4}$'),        # 英英-數數數數
            re.compile(r'^[A-Z]{3}\d{3}$'),        # 英英英-數數數
            re.compile(r'^\d{4}[A-Z]{2}$'),        # 數數數數-英英
            re.compile(r'^\d{4}[A-Z]\d$'),         # 數數數數-英數
            re.compile(r'[A-Z]{2}\d{4}$'),         # 英英-數數數數
            re.compile(r'^[A-Z]\d{5}$'),           # 英數-數數數數
            re.compile(r'^\d[A-Z]\d{4}$'),         # 數英-數數數數
            re.compile(r'\d{3}[A-Z]{2}$'),         # 數數數-英英
        ]
        if not any(pattern.match(text) for pattern in patterns):
            text = ''
    except IndexError:
        text = ''
    return text

# 初始化車輛和車牌偵測模型
car_model = YOLO(r"model\vehicle.pt")  # 車輛偵測模型
plate_model = YOLO(r'model\license_plate_detection.pt')  # 車牌偵測模型
thresh1 = 0.45  # 車輛偵測信心閾值
thresh2 = 0.45  # 車牌偵測信心閾值

# 定義車輛類別(機車、汽車、卡車、巴士皆標註為0)
v = [0]

# 所使用的文字辨識模型設定
PROCESSOR_CKPT = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(PROCESSOR_CKPT)
MODEL_NAME = "model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

distance_threshold = 122  # 設定距離閾值
previous_centers = {}  # 用來存放每個車輛的中心點資訊
car_id_counter = 0  # 初始化車輛ID計數器
max_lifespan = 10  # 設定車輛ID的最大壽命

# 設定影片路徑與讀取參數
path = r'video\input\video.mp4'
cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 初始化影片寫入參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'video\output\output.mp4', fourcc, fps, (width, height))

# 開始處理影片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 偵測車輛與車牌
    vehicles = car_model.predict(frame, conf=thresh1)[0]
    plates = plate_model.predict(frame, conf=thresh2)[0]

    detect_vehicle = []

    # 遍歷所有偵測到的車輛框
    for detect in vehicles.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = detect  # 取得車輛框的座標 (左上角和右下角) 以及置信度和類別
        if int(cls) in v:  # 如果偵測到的類別屬於指定車輛類別 (如機車、汽車、卡車、巴士等)
            detect_vehicle.append(([x1, y1, x2, y2], conf))  # 將車輛的座標和置信度加入車輛清單中
            print(f"Vehicle detected: {x1}, {y1}, {x2}, {y2}, {conf}, {cls}")

    # 遍歷所有偵測到的車牌框
    for plate in plates.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = plate  # 取得車牌框的座標 (左上角和右下角) 以及置信度和類別
        car = get_car([x1, y1, x2, y2, score], detect_vehicle)  # 檢查車牌是否位於任何已偵測到的車輛框內

        if car:  # 如果找到車輛框包含車牌框
            car_found = False  # 初始化變數，用於標記車輛是否被追蹤到
            for car_id, (prev_center, max_bbox, plate_text, lifespan) in previous_centers.items():
                current_center = calculate_center(x1, y1, x2, y2)  # 計算當前車牌框的中心點
                distance = calculate_distance(current_center, prev_center)  # 計算當前車牌框中心點與之前車牌框中心點的距離

                if distance < distance_threshold:  # 如果距離小於設定的閾值，則認為是同一車輛
                    car_found = True  # 標記為找到車輛
                    if (car[2] - car[0]) * (car[3] - car[1]) > max_bbox:  # 如果新的車輛框比之前儲存的更大，則更新
                        # 擷取車牌影像區域，增加邊界，避免邊緣信息丟失
                        license_plate_img = frame[max(0, int(y1 - 10)):min(height, int(y2 + 10)), max(0, int(x1 - 10)):min(width, int(x2 + 10))]
                        if license_plate_img.size > 0:
                            license_plate_img = cv2.resize(license_plate_img, (150, 50))  # 調整影像大小，適應OCR模型輸入
                            text = image_to_text(license_plate_img)  # 使用OCR模型將影像轉換為文字
                            if text != '':
                                previous_centers[car_id] = (current_center, (car[2] - car[0]) * (car[3] - car[1]), text, 0)  # 更新車輛信息
                                break
                    else:
                        # 如果新的車輛框不比之前儲存的更大，只重設壽命
                        center, bbox, text, lifespan = previous_centers[car_id]
                        previous_centers[car_id] = (center, bbox, text, 0)
                        break

            if not car_found:  # 如果沒有找到匹配的車輛，則認為是新車輛
                car_id_counter += 1  # 新車輛的ID
                car_id = car_id_counter
                current_center = calculate_center(x1, y1, x2, y2)  # 計算車輛中心點
                # 擷取新車輛的車牌影像區域
                license_plate_img = frame[max(0, int(y1 - 10)):min(height, int(y2 + 10)), max(0, int(x1 - 10)):min(width, int(x2 + 10))]
                if license_plate_img.size > 0:
                    license_plate_img = cv2.resize(license_plate_img, (150, 50))  # 調整影像大小
                    text = image_to_text(license_plate_img)  # 影像轉文字
                    if text != '':
                        previous_centers[car_id] = (current_center, (car[2] - car[0]) * (car[3] - car[1]), text, 0)  # 儲存車輛信息
                    else:
                        previous_centers[car_id] = (current_center, (car[2] - car[0]) * (car[3] - car[1]), 'detecting', 0)  # 沒有偵測到文字的情況下儲存為'detecting'

            # 在影像上繪製車牌框和車牌文字
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, previous_centers[car_id][2], (int(current_center[0]), int(current_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # 刪除過期的車輛ID
    keys_to_delete = []
    for car_id in previous_centers:
        center, bbox, text, lifespan = previous_centers[car_id]
        if lifespan >= max_lifespan:
            keys_to_delete.append(car_id)
        else:
            previous_centers[car_id] = (center, bbox, text, lifespan + 1)

    for key in keys_to_delete:
        del previous_centers[key]

    # 顯示及寫入影片
    cv2.imshow('Frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
