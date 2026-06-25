import cv2
import numpy as np
import os

query_path = "query.png"
dataset_dir = "images_folder/"

def process_image(image_path, size=(64,64)):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể nạp dữ liệu từ đường dẫn: {image_path}")
    
    img_resized = cv2.resize(img, size)

    print(f"[{image_path}] Kích thước ma trận ảnh xám", img_resized.shape)
    
    vector = img_resized.flatten()

    vector_normalized = vector / 255.0

    return vector_normalized

def l2_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

query_vector = process_image(query_path)

min_distance = float('inf')
best_match = None

print("\nBắt đầu quá trình quét tuyến tính trên tập dữ liệu")

for filename in os.listdir(dataset_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        db_image_path = os.path.join(dataset_dir, filename)
        
        try: 
            db_vector = process_image(db_image_path)

            distance = l2_distance(query_vector, db_vector)

            if distance < min_distance:
                min_distance = distance
                best_match = filename
        except ValueError as e:
            print(e)
 
print("\n--KẾT LUẬN HỆ THỐNG--")
if best_match:
    print(f"Tập tin có độ tương đồng cao nhất: {best_match}")
    print(f"Khoảng cách sai lệch tối thiểu ghi nhận: {min_distance:.4f}")
else:
    print("Hệ thống không tìm thấy dữ liệu hình ảnh phù hợp trong thư mục.")
