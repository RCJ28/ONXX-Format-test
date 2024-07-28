import onnxruntime as ort
import cv2
import numpy as np

def load_onnx_model(onnx_model_path):
    # Load ONNX model
    session = ort.InferenceSession(onnx_model_path)
    return session

def preprocess_image_onnx(image_path, img_size):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_resized = img_resized / 255.0  # Normalize to [0, 1]
    img_resized = np.transpose(img_resized, (2, 0, 1))  # Change from HWC to CHW
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    img_resized = img_resized.astype(np.float32)
    return img_resized

def draw_boxes(image, detections, classes):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{classes[int(cls)]}: {conf:.2f}"
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.rectangle(image, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, label_y - 7), font, font_scale, (0, 0, 0), thickness)
    return image


def predict_with_onnx(image_path, onnx_model_path, classes, img_size=640, conf_thresh=0.01, iou_thresh=0.1):
    # Load ONNX model
    session = load_onnx_model(onnx_model_path)
    
    print("model loaded")
    # Preprocess image
    img_input = preprocess_image_onnx(image_path, img_size)

    print("Preprocessing completed")
    print("image:" , img_input.shape)
    # Perform inference
    outputs = session.run(None, {"images":img_input})
    detections = np.array(outputs[0])
    print(outputs)
    # Filter detections by confidence threshold
    detections = detections[detections[:, 4] >= conf_thresh]

    
    #print("Model prediction done")
    # Apply non-maximum suppression
    keep = cv2.dnn.NMSBoxes(
        bboxes=detections[:, :4].tolist(),
        scores=detections[:, 4].tolist(),
        score_threshold=conf_thresh,
        nms_threshold=iou_thresh
    )
    detections = detections[keep]

    # Load image for drawing boxes
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes
    image_with_boxes = draw_boxes(image, detections, classes)

    return image_with_boxes

# Example usage
if __name__ == "__main__":
    
    image_path = "C:/Users/Jaya Chandra/Documents/GitHub/model-testing/test_image.jpg"
    onnx_model_path = "C:/Users/Jaya Chandra/Documents/GitHub/model-testing/best.onnx"
    classes = ['pothole']  # Replace with your actual class names

    result_image = predict_with_onnx(image_path, onnx_model_path, classes)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
