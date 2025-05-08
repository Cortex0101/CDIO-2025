import cv2
import numpy as np
import math

DEBUGGING = True  # Set False to disable the debug view

# size‐filter ratios (relative to the reference ball radius)
SIZE_RATIO_LOW  = 0.7
SIZE_RATIO_HIGH = 1.4

def adjust_gamma(image, gamma=1.5):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255
                      for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    g = adjust_gamma(eq, gamma=1.5)
    return cv2.GaussianBlur(g, (9, 9), 0)

def generate_mask_otsu(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, s = hsv[:,:,2], hsv[:,:,1]

    _, v_m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, s_m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.bitwise_and(v_m, s_m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m

def find_circle_candidates(mask):
    """
    Finds all bright low‐sat contours, fits each to a min‐enclosing circle,
    computes (arc_ratio, area_ratio), clusters into 2 groups, and returns
    candidates from the cluster closest to (1,1) in ratio‐space.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats, candidates = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:                   # skip tiny specks
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        if r < 5:
            continue

        perim     = cv2.arcLength(cnt, False)
        full_circ = 2 * math.pi * r
        arc_ratio = perim / full_circ

        circ_area  = math.pi * r * r
        area_ratio = area / circ_area

        feats.append([arc_ratio, area_ratio])
        candidates.append((int(x), int(y), int(r)))

    if not feats:
        return []

    data = np.array(feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    # pick the cluster whose center is closest to perfect circle (1,1)
    d0 = np.linalg.norm(centers[0] - np.array([1.0, 1.0]))
    d1 = np.linalg.norm(centers[1] - np.array([1.0, 1.0]))
    best = 0 if d0 < d1 else 1

    return [candidates[i] for i in range(len(candidates)) if labels[i][0] == best]

################
COLOR_RANGES = {
    "orange": {
        # tune these H/S/V ranges if your orange balls look different
        "lower": np.array([5, 100, 100], dtype=np.uint8),
        "upper": np.array([25, 255, 255], dtype=np.uint8),
    },
    # you can add e.g. "red", "green" here later...
}

def generate_mask(frame, color="white"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == "white":
        # exactly our Otsu + invert-Otsu on S approach
        v, s = hsv[:,:,2], hsv[:,:,1]
        _, v_m = cv2.threshold(v, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, s_m = cv2.threshold(s, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(v_m, s_m)

    elif color in COLOR_RANGES:
        lower = COLOR_RANGES[color]["lower"]
        upper = COLOR_RANGES[color]["upper"]
        mask = cv2.inRange(hsv, lower, upper)

    else:
        raise ValueError(f"Unsupported color ‘{color}’")

    # clean-up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

# … keep preprocess_frame(), adjust_gamma() exactly as before …

def find_circle_candidates(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats, cands = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30: continue
        (x,y), r = cv2.minEnclosingCircle(cnt)
        if r < 5: continue

        perim     = cv2.arcLength(cnt, False)
        arc_ratio = perim / (2*math.pi*r)
        area_ratio= area / (math.pi*r*r)

        feats.append([arc_ratio, area_ratio])
        cands.append((int(x),int(y),int(r)))

    if not feats:
        return []

    data = np.array(feats, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    d0 = np.linalg.norm(centers[0] - np.array([1.0,1.0]))
    d1 = np.linalg.norm(centers[1] - np.array([1.0,1.0]))
    best = 0 if d0 < d1 else 1

    return [cands[i] for i in range(len(cands)) if labels[i][0] == best]

def detect_balls(frame, color="white", debug=False):
    pre  = preprocess_frame(frame)
    mask = generate_mask(frame, color=color)
    cands= find_circle_candidates(mask)

    # size‐consistency filter against largest
    if cands:
        ref_r = max(cands, key=lambda c: c[2])[2]
        low, high = ref_r*SIZE_RATIO_LOW, ref_r*SIZE_RATIO_HIGH
        cands = [(x,y,r) for x,y,r in cands if low <= r <= high]

    centers = [(x,y) for x,y,_ in cands]

    if debug:
        views = [
            ("Original",     frame),
            ("Preprocessed", cv2.cvtColor(pre,  cv2.COLOR_GRAY2BGR)),
            ("Mask",         cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)),
            ("Detected",     frame.copy())
        ]
        for x,y,r in cands:
            cv2.circle(views[3][1], (x,y), r,   (0,255,0), 2)
            cv2.circle(views[3][1], (x,y),   2, (0,0,255), 3)

        idx = 0
        while True:
            title, img = views[idx]
            disp = img.copy()
            (w,h), _ = cv2.getTextSize(title,
                                      cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(disp,
                          (disp.shape[1]-w-20,10),
                          (disp.shape[1]-10,h+20),
                          (0,0,0),-1)
            cv2.putText(disp, title,
                        (disp.shape[1]-w-15,h+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

            cv2.imshow("Debug View", disp)
            key = cv2.waitKey(0) & 0xFF
            if   key == ord('d'): idx = (idx+1) % 4
            elif key == ord('a'): idx = (idx-1) % 4
            elif key == ord('q'): break

        cv2.destroyAllWindows()

    return centers

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        # switch to "orange" if you want orange‐ball detection
        balls = detect_balls(frame, color="orange", debug=DEBUGGING)
        print("Detected:", balls)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()