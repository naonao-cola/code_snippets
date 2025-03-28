import cv2

def draw_boxes(image, labels, class_names=None):
    if not labels:
        return image
    
    for label in labels:
        x, y, w, h, cls, conf = label
        x, y, w, h = int(x), int(y), int(w), int(h)
        x2, y2 = x + w, y + h
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        if class_names and cls in class_names:
            label_text = f"{class_names[cls]} {conf:.2f}"
        else:
            label_text = f"Class {cls} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(image, (x, y - text_height - baseline), (x + text_width, y), (0, 255, 0), -1)
        cv2.putText(image, label_text, (x, y - baseline), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return image