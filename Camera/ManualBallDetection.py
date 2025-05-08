import cv2
import numpy as np
import math

DEBUGGING = True

# size consistency
SIZE_RATIO_LOW, SIZE_RATIO_HIGH = 0.7, 1.4
# coverage thresholds
ARC_COVERAGE_THRESH   = 0.4
# no longer need COLOR_COVERAGE_THRESH

# calibration state
_stage = 0               # 0=white,1=orange,2=floor,3=wall,4=done
_color_order = ['white','orange']
_samples = {'white':[], 'orange':[]}
_floor_rect = None
_wall_line = None

# temp state
_adjusting   = False
_curr_center = None
_curr_radius = 30
_rect_clicks = []
_line_clicks = []

def _mouse_cb(evt, x, y, flags, img):
    global _adjusting, _curr_center, _curr_radius
    global _rect_clicks, _floor_rect, _line_clicks, _wall_line, _stage

    if _stage in (0,1):
        if evt == cv2.EVENT_LBUTTONDOWN and not _adjusting:
            _adjusting   = True
            _curr_center = (x, y)
            _curr_radius = 30

    elif _stage == 2 and evt == cv2.EVENT_LBUTTONDOWN:
        _rect_clicks.append((x,y))
        if len(_rect_clicks) == 2:
            (x0,y0),(x1,y1) = _rect_clicks
            _floor_rect = (min(x0,x1),min(y0,y1), max(x0,x1),max(y0,y1))
            _stage += 1

    elif _stage == 3 and evt == cv2.EVENT_LBUTTONDOWN:
        _line_clicks.append((x,y))
        if len(_line_clicks) == 2:
            _wall_line = (_line_clicks[0], _line_clicks[1])
            _stage += 1

def advance_stage():
    global _adjusting, _stage
    _adjusting = False
    _stage   += 1

def calibrate(image):
    """
    Click & size white balls (stage0), then orange (1),
    then two clicks for floor rect (2), two for wall line (3).
    Returns calib dict with stats and radius_range.
    """
    global _stage, _adjusting, _curr_center, _curr_radius

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", _mouse_cb, image)

    instr = {
      0: "WHITE balls: click center → UP/DOWN size → ENTER confirm → Q to skip",
      1: "ORANGE balls: click center → UP/DOWN size → ENTER confirm → Q to skip",
      2: "FLOOR: click 2 corners → Q to skip",
      3: "WALL: click 2 points → Q to skip"
    }

    while _stage < 4:
        disp = image.copy()
        h, w = disp.shape[:2]

        # draw samples
        for idx, cname in enumerate(_color_order):
            col = (255,255,255) if cname=='white' else (0,140,255)
            for (cx,cy),r in _samples[cname]:
                cv2.circle(disp,(cx,cy),r,col,2)

        # current circle
        if _stage in (0,1) and _adjusting and _curr_center:
            col = (255,255,255) if _stage==0 else (0,140,255)
            cv2.circle(disp,_curr_center,_curr_radius,col,2)

        # preview floor rect
        if _stage==2 and len(_rect_clicks)>=1:
            cv2.circle(disp, _rect_clicks[0], 5, (200,200,200), -1)
        if _stage==2 and len(_rect_clicks)==2:
            cv2.rectangle(disp, _rect_clicks[0], _rect_clicks[1], (200,200,200),2)

        # preview wall line
        if _stage==3 and len(_line_clicks)>=1:
            cv2.circle(disp, _line_clicks[0], 5, (200,200,200), -1)
        if _stage==3 and len(_line_clicks)==2:
            cv2.line(disp, _line_clicks[0], _line_clicks[1], (200,200,200),2)

        # banner
        cv2.rectangle(disp,(0,0),(w,40),(0,0,0),-1)
        cv2.putText(disp, instr[_stage], (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow("Calibration", disp)
        key = cv2.waitKey(1)&0xFF

        if _stage in (0,1):
            cname = _color_order[_stage]
            if _adjusting:
                if key in (13,10):
                    _samples[cname].append((_curr_center,_curr_radius))
                    _adjusting = False
                elif key==ord('q'):
                    advance_stage()
                elif key==ord('d'):
                    _curr_radius+=1
                elif key==ord('f'):
                    _curr_radius = max(5,_curr_radius-1)
            else:
                if key==ord('q'):
                    advance_stage()
        else:
            if key==ord('q'):
                _stage+=1

    cv2.destroyWindow("Calibration")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    calib = {}
    all_r = []

    # 1) white & orange stats
    for cname in _color_order:
        H,S,V = [],[],[]
        for (cx,cy),r in _samples[cname]:
            all_r.append(r)
            m = np.zeros(hsv.shape[:2],np.uint8)
            cv2.circle(m,(cx,cy),r,255,-1)
            H += hsv[:,:,0][m==255].tolist()
            S += hsv[:,:,1][m==255].tolist()
            V += hsv[:,:,2][m==255].tolist()

        arrH,arrS,arrV = np.array(H),np.array(S),np.array(V)
        calib[cname] = {
          'h_mean': float(arrH.mean()), 'h_std': float(arrH.std()),
          's_mean': float(arrS.mean()), 's_std': float(arrS.std()),
          'v_mean': float(arrV.mean()), 'v_std': float(arrV.std())
        }

    # 2) floor stats
    if _floor_rect:
        x0,y0,x1,y1 = _floor_rect
        roi = hsv[y0:y1, x0:x1]
        arrH,arrS,arrV = roi[:,:,0].ravel(), roi[:,:,1].ravel(), roi[:,:,2].ravel()
        calib['background'] = {
          'v_bg_mean': float(arrV.mean()), 'v_bg_std': float(arrV.std())
        }

    # 3) wall stats
    if _wall_line:
        m = np.zeros(hsv.shape[:2],np.uint8)
        cv2.line(m, _wall_line[0], _wall_line[1],255,3)
        vals = hsv[m==255]
        arrH,arrS = vals[:,0], vals[:,1]
        calib['wall'] = {
          'h_w_mean': float(arrH.mean()), 'h_w_std': float(arrH.std()),
          's_w_mean': float(arrS.mean()), 's_w_std': float(arrS.std())
        }

    # 4) radius range
    mn, sd = float(np.mean(all_r)), float(np.std(all_r))
    calib['radius_range'] = ( max(5,int(mn-2*sd)), int(mn+2*sd) )

    return calib

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0,(8,8))
    eq = clahe.apply(gray)
    table = np.array([(i/255.0)**(1/1.5)*255 for i in range(256)],np.uint8)
    gm = cv2.LUT(eq,table)
    return cv2.GaussianBlur(gm,(9,9),0)

def detect_balls(frame, calib, debug=False):
    pre   = preprocess_frame(frame)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(pre,50,150)

    raw = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,
                          dp=1.2, minDist=30,
                          param1=50, param2=30,
                          minRadius=calib['radius_range'][0],
                          maxRadius=calib['radius_range'][1])

    cands = []
    h, w = hsv.shape[:2]
    Y, X = np.ogrid[:h,:w]

    # loop circles
    if raw is not None:
        for x,y,r in np.uint16(np.round(raw[0])):
            # arc‐coverage
            ring = ((X-x)**2 + (Y-y)**2 >= (r-2)**2) & ((X-x)**2 + (Y-y)**2 <= (r+2)**2)
            arc_frac = np.count_nonzero(edges[ring]) / float(np.count_nonzero(ring))
            if arc_frac < ARC_COVERAGE_THRESH: continue

            # mean HSV inside circle
            circ = (X-x)**2 + (Y-y)**2 <= r*r
            vals = hsv[circ]
            meanH, meanS, meanV = float(vals[:,0].mean()), float(vals[:,1].mean()), float(vals[:,2].mean())

            # background cut
            if 'background' in calib:
                bg = calib['background']
                if meanV <= bg['v_bg_mean'] + bg['v_bg_std']:
                    continue

            # wall cut
            if 'wall' in calib:
                wl = calib['wall']
                if abs(meanH - wl['h_w_mean']) <= wl['h_w_std']*2 and abs(meanS - wl['s_w_mean']) <= wl['s_w_std']*2:
                    continue

            # white test
            wht = calib['white']
            if meanV >= wht['v_mean'] - wht['v_std'] and meanS <= wht['s_mean'] + wht['s_std']:
                cands.append((int(x),int(y),int(r)))
                continue

            # orange test
            org = calib['orange']
            if abs(meanH - org['h_mean']) <= 2*org['h_std'] and meanS >= org['s_mean'] - org['s_std']:
                cands.append((int(x),int(y),int(r)))
                continue

    # size consistency
    if cands:
        ref_r = max(c[2] for c in cands)
        lo,hi = ref_r*SIZE_RATIO_LOW, ref_r*SIZE_RATIO_HIGH
        cands = [c for c in cands if lo <= c[2] <= hi]

    centers = [(x,y) for x,y,_ in cands]

    if debug:
        views = [
            ("Original", frame),
            ("Edges",    cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)),
            ("Detected", frame.copy())
        ]
        for x,y,r in cands:
            cv2.circle(views[2][1],(x,y),r,(0,255,0),2)
            cv2.circle(views[2][1],(x,y),2,(0,0,255),3)

        idx = 0
        while True:
            title,img = views[idx]
            disp = img.copy()
            (tw,th),_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(disp,(w-tw-20,10),(w-10,th+20),(0,0,0),-1)
            cv2.putText(disp,title,(w-tw-15,th+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.imshow("Debug View",disp)
            k = cv2.waitKey(0)&0xFF
            if   k==ord('d'): idx=(idx+1)%len(views)
            elif k==ord('a'): idx=(idx-1)%len(views)
            elif k==ord('q'): break
        cv2.destroyAllWindows()

    return centers

if __name__=="__main__":
    cap = cv2.VideoCapture(1)
    ret, sample = cap.read()
    if not ret:
        raise RuntimeError("Camera read failed")
    calib = calibrate(sample)

    while True:
        ret, frame = cap.read()
        if not ret: break
        balls = detect_balls(frame, calib, debug=DEBUGGING)
        print("Detected centers:", balls)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
