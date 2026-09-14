
import cv2
from ultralytics import YOLO

# ====== CẤU HÌNH ======
MODEL_PATH = "best.pt"     # Đường dẫn tới file model đã train (đặt cùng thư mục hoặc sửa đường dẫn đầy đủ)
CAMERA_INDEX = 0           # 0 = camera mặc định của laptop
CONF_THRESHOLD = 0.5       # Ngưỡng độ tin cậy, có thể chỉnh lại tùy độ chính xác model
# ========================


def main():
    # Bước A: Load model YOLOv8 đã train
    model = YOLO(MODEL_PATH)

    # Bước B: Mở camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Không mở được camera. Kiểm tra lại CAMERA_INDEX hoặc quyền truy cập camera.")
        return

    print("Nhấn phím 'q' để thoát chương trình.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ camera.")
            break

        # Bước C: Đưa frame vào model để predict
        # verbose=False để không in log rối terminal mỗi frame
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)

        # results[0].boxes chứa danh sách các box phát hiện được trong frame này
        boxes = results[0].boxes

        if len(boxes) > 0:
            # Nếu có nhiều tay được phát hiện, lấy box có độ tin cậy cao nhất
            best_box = max(boxes, key=lambda b: float(b.conf[0]))

            # Tọa độ góc trên-trái và dưới-phải (pixel thực tế trên frame)
            x1, y1, x2, y2 = best_box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Tọa độ tâm và kích thước box (pixel) — dữ liệu cần cho việc tính khoảng cách sau này
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_width = x2 - x1
            box_height = y2 - y1

            confidence = float(best_box.conf[0])

            # Bước D: In tọa độ ra terminal (bạn có thể thay bằng cách lưu vào biến/gửi đi nơi khác)
            print(
                f"Tay phát hiện | Tâm: ({cx}, {cy}) | "
                f"Kích thước box: {box_width}x{box_height} px | "
                f"Conf: {confidence:.2f}"
            )

            # Bước E: Vẽ khung và thông tin lên frame để xem trực quan
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            label = f"({cx},{cy}) {box_width}x{box_height}px conf:{confidence:.2f}"
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # Hiển thị frame
        cv2.imshow("Hand Detection - YOLOv8", frame)

        # Bước F: Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
