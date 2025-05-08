import cv2
import numpy as np
import math

DEBUGGING = True            # turn off for headless/deployment
SIZE_RATIO_LOW, SIZE_RATIO_HIGH = 0.7, 1.4

def adjust_gamma(image, gamma=1.5):
    inv = 1.0/gamma
    table = np.array([(i/255.0)**inv * 255 for i in range(256)], dtype="uint8")
    return cv2.LUT(image, table)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq   = clahe.apply(gray)
    g    = adjust_gamma(eq, gamma=1.5)
    return cv2.GaussianBlur(g, (9,9), 0)

def generate_mask_otsu(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v, s = hsv[:,:,2], hsv[:,:,1]

    _, v_m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, s_m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    m = cv2.bitwise_and(v_m, s_m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m

def find_circle_candidates(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats, cands = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:            # drop tiny specks
            continue

        (x,y), r = cv2.minEnclosingCircle(cnt)
        if r < 5:                # drop micros
            continue

        perim     = cv2.arcLength(cnt, False)
        arc_ratio = perim / (2*math.pi*r)
        area_ratio= area / (math.pi*r*r)

        feats.append([arc_ratio, area_ratio])
        cands.append((int(x),int(y),int(r)))

    if not feats:
        return []

    data = np.array(feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # pick cluster whose center is closest to (1,1)
    d0 = np.linalg.norm(centers[0] - np.array([1.0,1.0]))
    d1 = np.linalg.norm(centers[1] - np.array([1.0,1.0]))
    best = 0 if d0 < d1 else 1

    return [cands[i] for i in range(len(cands)) if labels[i][0]==best]

def detect_balls(frame, color="white", debug=False):
    """
    1) preprocess → gray, CLAHE, gamma, blur
    2) edges = Canny(preprocessed)
    3) run HoughCircles on edges to get ALL round candidates
    4) build a circular mask for each (x,y,r) and check that
       at least, say, 30% of its pixels are 'white' (or orange)
       by looking at our existing color‐mask.
    5) size‐consistency filter (optional)
    """
    # 1) preprocess & color-mask
    pre  = preprocess_frame(frame)
    mask = generate_mask_otsu(frame)

    # 2) detect *all* circles via edges
    edges = cv2.Canny(pre, 50, 150)
    raw = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=30,
        param1=50, param2=30,
        minRadius=5,  maxRadius=200
    )

    candidates = []
    if raw is not None:
        for x, y, r in np.uint16(np.around(raw[0])):
            # build a boolean mask of this circle
            Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
            circle_area = (X - x)**2 + (Y - y)**2 <= r*r

            # compute what fraction of that circle is 'white'
            white_frac = np.count_nonzero(mask[circle_area]) / float(circle_area.sum())
            if white_frac < 0.3:
                continue

            candidates.append((int(x), int(y), int(r)))

    # 3) size‐consistency: only keep circles close to the largest
    if candidates:
        ref_r = max(c[2] for c in candidates)
        lo, hi = ref_r * 0.7, ref_r * 1.4
        candidates = [c for c in candidates if lo <= c[2] <= hi]

    centers = [(x, y) for x, y, _ in candidates]

    # 4) debug window (A/D to cycle Original, Pre, Mask, Detected)
    if debug:
        views = [
            ("Original",     frame),
            ("Preprocessed", cv2.cvtColor(pre,  cv2.COLOR_GRAY2BGR)),
            ("Mask",         cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)),
            ("Detected",     frame.copy())
        ]
        for x, y, r in candidates:
            cv2.circle(views[3][1], (x, y), r,   (0,255,0), 2)
            cv2.circle(views[3][1], (x, y),   2, (0,0,255), 3)

        idx = 0
        while True:
            title, img = views[idx]
            disp = img.copy()
            (w, h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(disp,
                          (disp.shape[1]-w-20, 10),
                          (disp.shape[1]-10,  h+20),
                          (0,0,0), -1)
            cv2.putText(disp, title,
                        (disp.shape[1]-w-15, h+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Debug View", disp)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('d'):
                idx = (idx + 1) % len(views)
            elif key == ord('a'):
                idx = (idx - 1) % len(views)
            elif key == ord('q'):
                break
        cv2.destroyAllWindows()

    return centers

if __name__=="__main__":
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret: break

        balls = detect_balls(frame, debug=DEBUGGING)
        print("Detected balls:", balls)

        if cv2.waitKey(1)&0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
