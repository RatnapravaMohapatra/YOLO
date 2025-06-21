import cv2
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('../Yolo-weights/yolov8n.pt')

# Run inference on an image
# Do NOT use show=True here, as we'll display with OpenCV
results = model('Images/3.jpg')

# Process the results
for r in results:
    # r.plot() returns the image with detections drawn on it as a NumPy array
    im_array = r.plot()
    # Convert from RGB (Ultralytics default) to BGR (OpenCV default) if needed
    # Ultralytics often returns BGR directly for cv2.imshow compatibility, but good to be aware.
    # If colors look off, uncomment the line below.
    # im_array = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)

    cv2.imshow("YOLO Detections", im_array)
    # Wait for a key press indefinitely. The window will stay open until a key is pressed.
    cv2.waitKey(0)

cv2.destroyAllWindows() # Close all OpenCV windows when done