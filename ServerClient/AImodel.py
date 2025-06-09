from ultralytics import YOLO
import cv2
import numpy as np

class AIModel:
    def __init__(self):
        self.model = YOLO("ball_detect/v7/weights/best.pt")
        self.colors = {
            "egg": (255, 0, 0),      # Red
            "robot": (0, 255, 0),    # Green
            "white_ball": (255, 255, 255),  # White
            "orange_ball": (0, 165, 255),   # Orange
            "small_goal": (255, 0, 0),  # Blue
            "large_goal": (255, 255, 0), # Yellow
            "wall": (128, 128, 128),    # Gray
            "cross": (0, 0, 145),       # Dark Blue
        }

    def predict(self, frame):
        results = self.model.predict(source=frame, conf=0.3, iou=0.5)
        return results[0] if results else None

    def show_results(self, frame, options={"boxes": False, "masks": False, "conf": True, "labels": True, "center": True}):
        results = self.predict(frame)
        if results is None:
            return frame

        img = frame.copy()
        # Draw boxes and centers
        if options.get("boxes", True):
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().item()
                cls = box.cls[0].item()
                label = results.names[cls] if options.get("labels", True) else ""
                color = self.colors.get(label, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                if options.get("conf", True):
                    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if options.get("center", True):
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(img, f"({cx},{cy})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw masks
        if options.get("masks", True) and results.masks is not None:
            for polygon in results.masks.xy:
                pts = np.array(polygon, dtype=np.int32)
                # use class-based color if available
                cls = results.boxes.cls[0].item()
                color = self.colors.get(results.names[cls], (0, 0, 255))
                cv2.fillPoly(img, [pts], color)

        return img

    def get_objects_info(self,
                         frame: np.ndarray,
                         classes_to_keep: list[str] | None = None,
                         return_bbox: bool = True,
                         return_mask: bool = False,
                         return_centroid: bool = True
                         ) -> list[dict]:
        """
        Detects objects in `frame` and returns a list of info dicts.

        Args:
            frame: BGR image.
            classes_to_keep: list of class names to include (None = all).
            return_bbox: include 'bbox': [x1,y1,x2,y2].
            return_mask: include 'mask': binary mask array.
            return_centroid: include 'centroid': (cx, cy).

        Returns:
            A list of dicts, e.g.:
            [
                {
                    'class': 'robot',
                    'confidence': 0.87,
                    'bbox': [100, 150, 200, 250],
                    'centroid': (150.0, 200.0),
                    'mask': np.ndarray  # if return_mask=True
                },
                …
            ]
        """
        results = self.predict(frame)
        if results is None:
            return []

        # pre‐extract masks if needed
        masks = None
        if return_mask and results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # shape: (N, H, W)

        infos = []
        for idx, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item())
            cls_name = results.names[cls_id]
            if classes_to_keep and cls_name not in classes_to_keep:
                continue

            info = {
                'class': cls_name,
                'confidence': float(box.conf[0].item())
            }

            # Bounding box
            if return_bbox:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                info['bbox'] = [int(x1), int(y1), int(x2), int(y2)]

            # Centroid
            if return_centroid:
                if return_bbox:
                    cx = (info['bbox'][0] + info['bbox'][2]) / 2.0
                    cy = (info['bbox'][1] + info['bbox'][3]) / 2.0
                else:
                    # compute from mask
                    if masks is None:
                        masks = results.masks.data.cpu().numpy()
                    mask = masks[idx]
                    cnts, _ = cv2.findContours(
                        (mask > 0).astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_NONE
                    )
                    if cnts:
                        M = cv2.moments(cnts[0])
                        if M['m00'] != 0:
                            cx = M['m10'] / M['m00']
                            cy = M['m01'] / M['m00']
                        else:
                            cx, cy = None, None
                    else:
                        cx, cy = None, None
                info['centroid'] = (float(cx), float(cy) if cy is not None else None)

            # Mask
            if return_mask:
                info['mask'] = masks[idx]

            infos.append(info)

        return infos

    def find_closest_ball(self, frame: np.ndarray, ball_color: str = "white") -> dict | None:
        """
        Finds the closest ball of the specified color in the frame.

        Args:
            frame: BGR image.
            ball_color: Color of the ball to find (e.g., "white", "orange").

        Returns:
            A dict with 'class', 'confidence', 'bbox', 'centroid' if found, else None.
        """
        results = self.predict(frame)
        if results is None or results.boxes is None:
            return None

        closest_ball = None
        min_distance = float('inf')

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = results.names[cls_id]
            if cls_name != ball_color:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            distance = np.sqrt(cx**2 + cy**2)

            if distance < min_distance:
                min_distance = distance
                closest_ball = {
                    'class': cls_name,
                    'confidence': float(box.conf[0].item()),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'centroid': (float(cx), float(cy))
                }

        return closest_ball

    @staticmethod
    def masks_intersect(mask1: np.ndarray, mask2: np.ndarray) -> bool:
        """
        Returns True if two binary masks intersect (share any common 'True'/non-zero pixels).

        Args:
            mask1: 2D array of zeros and ones (or boolean).
            mask2: same shape as mask1.
        """
        if mask1.shape != mask2.shape:
            raise ValueError("Mask shapes must match for intersection test")
        # Convert to boolean and test overlap
        return np.logical_and(mask1.astype(bool), mask2.astype(bool)).any()

if __name__ == "__main__":
    model = AIModel()
    img = cv2.imread("AI/images/image_0.jpg")

    results_img = model.show_results(img, options={
        "boxes": True,
        "labels": True,
        "center": True,
        "conf": False,
        "masks": False
    })

    closest_ball = model.find_closest_ball(img, ball_color="white")
    if closest_ball:
        print(f"Closest ball found: {closest_ball}")
    else:
        print("No closest ball found.")

    # draw the closest ball on the image
    if closest_ball:
        x1, y1, x2, y2 = closest_ball['bbox']
        cv2.rectangle(results_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = closest_ball['centroid']
        cv2.circle(results_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(results_img, f"{closest_ball['class']} {closest_ball['confidence']:.2f}",
                    (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Detection Results", results_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    info = model.get_objects_info(img,
                                    classes_to_keep=["robot","white"],
                                    return_bbox=False,
                                    return_mask=False)
    for obj in info:
        print(obj)
