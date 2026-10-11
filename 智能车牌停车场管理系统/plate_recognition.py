import cv2
import numpy as np
import os


# =====================================================
# 基本配置
# =====================================================
TEMPLATE_ROOT = "plate_template_24x48_done"

IMAGE_PATHS = [
    "test.jpg",
    "test1.jpg",
    "test2.jpg",
    "test3.jpg",
    "test4.jpg"
]

# True：显示定位、二值化、绿色字符切割框
# False：只输出识别结果
SHOW_PROCESS = True


# =====================================================
# 中文路径读取图片
# =====================================================
def cv_imread(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)


# =====================================================
# 显示窗口缩放
# =====================================================
def resize_show(img, max_width=900):
    h, w = img.shape[:2]
    if w <= max_width:
        return img

    scale = max_width / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


# =====================================================
# 字符归一化为 24 × 48
# 输入必须是白字黑底
# =====================================================
def resize_and_center(binary):
    if binary is None or binary.size == 0:
        return np.zeros((48, 24), dtype=np.uint8)

    if len(binary.shape) == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(binary == 255)

    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((48, 24), dtype=np.uint8)

    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)

    char = binary[y1:y2 + 1, x1:x2 + 1]

    h, w = char.shape

    target_w = 24
    target_h = 48

    scale = min((target_w - 4) / w, (target_h - 4) / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    char = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = char

    return canvas


# =====================================================
# 模板归一化
# 模板已经是白字黑底，所以这里绝对不反色
# =====================================================
def normalize_template(img):
    if img is None or img.size == 0:
        return np.zeros((48, 24), dtype=np.uint8)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    return resize_and_center(binary)


# =====================================================
# 加载模板
# 文件夹结构：
# plate_template_24x48_done/
#   chinese/
#   letter/
#   number/
# =====================================================
def load_templates(template_root):
    templates = {
        "chinese": {},
        "letter": {},
        "number": {}
    }

    for category in templates:
        folder = os.path.join(template_root, category)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"找不到模板文件夹：{folder}")

        for file in os.listdir(folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            label = os.path.splitext(file)[0]

            if category in ["letter", "number"]:
                label = label.upper()

            path = os.path.join(folder, file)
            img = cv_imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("模板读取失败：", path)
                continue

            templates[category][label] = normalize_template(img)

    print("模板加载完成：")
    print("汉字模板数量：", len(templates["chinese"]))
    print("字母模板数量：", len(templates["letter"]))
    print("数字模板数量：", len(templates["number"]))

    return templates


# =====================================================
# 车牌定位：HSV 蓝色匹配 + 轮廓筛选
# =====================================================
def locate_blue_plate(image):
    img_h, img_w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_ranges = [
        (np.array([95, 80, 80]), np.array([130, 255, 255])),
        (np.array([90, 70, 60]), np.array([135, 255, 255])),
        (np.array([90, 60, 50]), np.array([140, 255, 255]))
    ]

    best_candidate = None
    best_mask = None

    for lower_blue, upper_blue in blue_ranges:
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        blue_mask[:int(img_h * 0.20), :] = 0

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            blue_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if w <= 0 or h <= 0:
                continue

            ratio = w / h
            rect_area = w * h

            if ratio < 2.0 or ratio > 5.8:
                continue

            if w < img_w * 0.12:
                continue

            if h < img_h * 0.04:
                continue

            if rect_area < img_w * img_h * 0.003:
                continue

            if rect_area > img_w * img_h * 0.18:
                continue

            if w > img_w * 0.85:
                continue

            if h > img_h * 0.40:
                continue

            candidate = image[y:y + h, x:x + w]

            if candidate.size == 0:
                continue

            candidate_hsv = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)

            candidate_blue = cv2.inRange(candidate_hsv, lower_blue, upper_blue)
            blue_ratio = np.sum(candidate_blue == 255) / candidate_blue.size

            lower_white = np.array([0, 0, 140])
            upper_white = np.array([180, 120, 255])
            candidate_white = cv2.inRange(candidate_hsv, lower_white, upper_white)
            white_pixels = np.sum(candidate_white == 255)

            if blue_ratio < 0.25:
                continue

            if white_pixels < 50:
                continue

            ratio_score = 1.0 - abs(ratio - 3.2) / 3.2
            ratio_score = max(0, ratio_score)

            area_score = min(rect_area / (img_w * img_h * 0.06), 1.0)
            blue_score = min(blue_ratio / 0.75, 1.0)
            white_score = min(white_pixels / 1800, 1.0)
            position_score = (y + h / 2) / img_h

            score = (
                0.35 * ratio_score +
                0.25 * area_score +
                0.25 * blue_score +
                0.10 * white_score +
                0.05 * position_score
            )

            candidates.append((score, x, y, w, h))

        if candidates:
            candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
            best_candidate = candidates[0]
            best_mask = blue_mask
            break

    if best_candidate is None:
        return None, None, None

    score, x, y, w, h = best_candidate

    pad_x = int(w * 0.04)
    pad_y = int(h * 0.12)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    plate = image[y1:y2, x1:x2]
    box = (x1, y1, x2, y2)

    return plate, best_mask, box


# =====================================================
# 只去除车牌矩形白色边框线
# 不删除车牌内部字符，避免误删 I / 1
# =====================================================
def remove_plate_frame_lines(binary):
    """
    安全去除车牌白色矩形边框线。
    重点修改：
    1. 不再在左侧 12% 区域做大面积竖线开运算；
       否则“苏、浙、沪、冀”等汉字的竖笔画会被当成边框删掉。
    2. 只删除“很长的横线”和“很高的竖线”。
    3. 不再用连通域删除靠左的窄高区域，保护第一个汉字。
    """
    h, w = binary.shape

    line_mask = np.zeros_like(binary)

    # ---------- 上下边框：只识别非常长的横线 ----------
    top_h = int(h * 0.24)
    bottom_start = int(h * 0.76)

    close_h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, int(w * 0.035)), 1)
    )

    open_h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(80, int(w * 0.50)), 1)
    )

    if top_h > 0:
        top_band = binary[:top_h, :]
        top_close = cv2.morphologyEx(top_band, cv2.MORPH_CLOSE, close_h_kernel)
        top_line = cv2.morphologyEx(top_close, cv2.MORPH_OPEN, open_h_kernel)
        line_mask[:top_h, :] = cv2.bitwise_or(line_mask[:top_h, :], top_line)

    if bottom_start < h:
        bottom_band = binary[bottom_start:, :]
        bottom_close = cv2.morphologyEx(bottom_band, cv2.MORPH_CLOSE, close_h_kernel)
        bottom_line = cv2.morphologyEx(bottom_close, cv2.MORPH_OPEN, open_h_kernel)
        line_mask[bottom_start:, :] = cv2.bitwise_or(
            line_mask[bottom_start:, :],
            bottom_line
        )

    # ---------- 左右边框：只识别非常高的竖线 ----------
    # 注意：这里不能把左侧整块区域都清掉，否则第一个汉字会被删。
    side_w = int(w * 0.18)

    close_v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(6, int(h * 0.035)))
    )

    open_v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(55, int(h * 0.68)))
    )

    if side_w > 0:
        left_band = binary[:, :side_w]
        left_close = cv2.morphologyEx(left_band, cv2.MORPH_CLOSE, close_v_kernel)
        left_line = cv2.morphologyEx(left_close, cv2.MORPH_OPEN, open_v_kernel)
        line_mask[:, :side_w] = cv2.bitwise_or(line_mask[:, :side_w], left_line)

        right_band = binary[:, w - side_w:]
        right_close = cv2.morphologyEx(right_band, cv2.MORPH_CLOSE, close_v_kernel)
        right_line = cv2.morphologyEx(right_close, cv2.MORPH_OPEN, open_v_kernel)
        line_mask[:, w - side_w:] = cv2.bitwise_or(
            line_mask[:, w - side_w:],
            right_line
        )

    # 轻微膨胀一下边框 mask，但不能太大，否则会吃掉汉字笔画
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_mask = cv2.dilate(line_mask, small_kernel, iterations=1)

    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))

    return cleaned


# =====================================================
# 去除螺丝点
# 只删除左右螺丝区域的小圆点，不删除内部字符
# =====================================================
def _safe_erase_circle(binary, center_x, center_y, radius):
    """
    在二值图上擦掉一个螺丝圆点区域。
    用椭圆形 mask，比直接删矩形更不容易误伤字符笔画。
    """
    clean = binary.copy()
    h, w = clean.shape

    cx = int(round(center_x))
    cy = int(round(center_y))
    r = int(round(radius))

    if r <= 1:
        return clean

    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    mask = np.zeros_like(clean)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    # 稍微向外扩一点，清掉螺丝边缘反光残留
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    clean[mask == 255] = 0
    return clean


def _remove_separate_screw_components(binary):
    """
    删除已经和字符分离的小螺丝点。
    只在常见螺丝区域处理：上边/下边 + 左右安装孔附近。
    """
    h, w = binary.shape
    clean = binary.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    # 常见蓝牌螺丝大致在左右两侧，上下边附近。
    # 这里给的是区域，不是固定点；不同图片裁剪略偏也能覆盖。
    screw_x_zones = [
        (0.10, 0.36),
        (0.58, 0.90),
    ]

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        if cw <= 0 or ch <= 0 or area <= 0:
            continue

        cx = (x + cw / 2) / w
        cy = (y + ch / 2) / h

        in_x_zone = any(a <= cx <= b for a, b in screw_x_zones)
        in_y_zone = (0.03 <= cy <= 0.32) or (0.68 <= cy <= 0.97)

        if not (in_x_zone and in_y_zone):
            continue

        ratio = cw / ch if ch != 0 else 0
        fill_ratio = area / (cw * ch + 1e-6)

        # 螺丝点通常是小而紧凑的圆/椭圆；字符主体一般更高、更长。
        is_small_enough = 8 <= area <= h * w * 0.010
        is_compact = 0.42 <= ratio <= 2.40 and fill_ratio >= 0.18
        not_tall_stroke = ch <= h * 0.24 and cw <= w * 0.090

        if is_small_enough and is_compact and not_tall_stroke:
            clean[labels == i] = 0

    return clean


def _remove_screw_by_local_circle(binary):
    """
    删除和字符轻微粘连的螺丝干扰。

    思路：
    1. 只在螺丝可能出现的局部窗口搜索；
    2. 用 HoughCircles 或局部白点质心找圆形亮斑；
    3. 只擦掉半径很小的圆形区域，避免删除整段字符。
    """
    h, w = binary.shape
    clean = binary.copy()

    # 如果你的数据集螺丝位置不同，优先改这里。
    # 格式：(x_ratio, y_ratio)，相对于 420×140 归一化后的车牌二值图。
    expected_centers = [
        (0.22, 0.17),
        (0.78, 0.17),
        (0.22, 0.83),
        (0.78, 0.83),
    ]

    min_r = max(3, int(h * 0.030))
    max_r = max(min_r + 2, int(h * 0.055))

    for xr, yr in expected_centers:
        cx0 = int(w * xr)
        cy0 = int(h * yr)

        rx = int(w * 0.145)
        ry = int(h * 0.210)

        x1 = max(0, cx0 - rx)
        x2 = min(w, cx0 + rx)
        y1 = max(0, cy0 - ry)
        y2 = min(h, cy0 + ry)

        patch = clean[y1:y2, x1:x2]

        if patch.size == 0 or np.sum(patch == 255) < 8:
            continue

        # 先尝试 HoughCircles，能处理部分和字符粘连的圆形螺丝。
        blur = cv2.medianBlur(patch, 5)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(6, int(h * 0.08)),
            param1=80,
            param2=7,
            minRadius=min_r,
            maxRadius=max_r
        )

        circle_candidates = []

        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for pcx, pcy, pr in circles:
                gcx = x1 + pcx
                gcy = y1 + pcy
                dist_to_expected = ((gcx - cx0) ** 2 + (gcy - cy0) ** 2) ** 0.5

                # 不能离安装孔预期位置太远，否则容易误检成 8/9 的环。
                if dist_to_expected > max(rx, ry) * 0.58:
                    continue

                # 候选圆心必须真的靠近上下安装孔，不能落在字符主体区。
                if yr < 0.50 and (gcy / h) > 0.225:
                    continue
                if yr > 0.50 and (gcy / h) < 0.775:
                    continue

                mask = np.zeros_like(clean)
                cv2.circle(mask, (gcx, gcy), int(pr), 255, -1)
                inside_white = np.sum((clean == 255) & (mask == 255))
                circle_area = max(1, np.sum(mask == 255))
                white_ratio = inside_white / circle_area

                # 螺丝点通常是比较实心的亮斑；8/9/0 的环形笔画 white_ratio 较低。
                if 0.35 <= white_ratio <= 0.95:
                    circle_candidates.append((gcx, gcy, pr, white_ratio))

        # Hough 找不到时，退回到“局部小白块质心”。
        if not circle_candidates:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                patch,
                connectivity=8
            )

            for i in range(1, num_labels):
                px, py, pcw, pch, area = stats[i]
                if pcw <= 0 or pch <= 0 or area <= 0:
                    continue

                ratio = pcw / pch if pch != 0 else 0
                fill_ratio = area / (pcw * pch + 1e-6)

                # 这里比 separate component 更宽松一些，因为螺丝可能被字符粘了一小段。
                if not (10 <= area <= h * w * 0.018):
                    continue
                if not (0.35 <= ratio <= 2.80):
                    continue
                if fill_ratio < 0.12:
                    continue
                if pcw > w * 0.11 or pch > h * 0.28:
                    continue

                pcx, pcy = centroids[i]
                gcx = x1 + pcx
                gcy = y1 + pcy
                # 质心候选也必须靠近上下安装孔，避免把数字 8/9 的局部误删。
                if yr < 0.50 and (gcy / h) > 0.225:
                    continue
                if yr > 0.50 and (gcy / h) < 0.775:
                    continue

                pr = int(max(pcw, pch) * 0.62)
                pr = max(min_r, min(max_r, pr))

                circle_candidates.append((gcx, gcy, pr, fill_ratio))

        if not circle_candidates:
            continue

        # 每个安装孔窗口最多删一个最像螺丝的圆，避免误删多个字符笔画。
        circle_candidates = sorted(
            circle_candidates,
            key=lambda item: abs(item[0] - cx0) + abs(item[1] - cy0) - 6 * item[3]
        )

        gcx, gcy, gr, _ = circle_candidates[0]
        clean = _safe_erase_circle(clean, gcx, gcy, max(min_r, gr))

    return clean


def remove_screw_noise(binary):
    """
    去除车牌螺丝干扰。

    这一版分两步：
    1. 删除已经独立成块的小圆点；
    2. 在常见安装孔位置，用局部圆形检测删除和字符轻微粘连的螺丝。

    注意：不要在整张车牌上随便删圆形，否则 8、9、0 的圆环会被误删。
    """
    if binary is None or binary.size == 0:
        return binary

    clean = binary.copy()
    clean = _remove_separate_screw_components(clean)
    clean = _remove_screw_by_local_circle(clean)
    clean = _remove_separate_screw_components(clean)

    return clean


# =====================================================
# 单字符槽内去螺丝/铆钉残留
# =====================================================
def remove_screw_noise_in_slot(slot_img, idx):
    """
    字符已经切成槽以后，再做一次小圆点清理。
    这样可以处理：螺丝点被切进某一个字符槽，导致模板匹配偏移。
    """
    if slot_img is None or slot_img.size == 0:
        return slot_img

    # 第一个汉字笔画碎片很多，不在槽内额外删点，避免误删“沪/浙/苏”等偏旁。
    if idx == 0:
        return slot_img

    clean = slot_img.copy()
    h, w = clean.shape
    total_white = max(1, np.sum(clean == 255))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        if cw <= 0 or ch <= 0 or area <= 0:
            continue

        cx = (x + cw / 2) / w
        cy = (y + ch / 2) / h
        ratio = cw / ch if ch != 0 else 0
        fill_ratio = area / (cw * ch + 1e-6)

        # 螺丝通常在字符槽的上/下边缘附近，且是次要小块。
        in_screw_y = cy < 0.26 or cy > 0.74
        compact = 0.45 <= ratio <= 2.35 and fill_ratio >= 0.18
        small = 6 <= area <= max(12, h * w * 0.12)
        minor = area / total_white <= 0.30

        # 不删除太高的竖笔画，避免误删 1/I/K/R 的主体。
        not_character_stem = ch <= h * 0.33 and cw <= w * 0.45

        if in_screw_y and compact and small and minor and not_character_stem:
            clean[labels == i] = 0

    return clean


# =====================================================
# 去除小噪声，但保留 I / 1 这种细长字符
# =====================================================
def remove_small_noise_keep_thin_chars(binary):
    h, w = binary.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    clean = np.zeros_like(binary)

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        if area < 8:
            continue

        if ch < h * 0.12 and area < 25:
            continue

        clean[labels == i] = 255

    return clean




# =====================================================
# 更精确删除车牌外框残留：只处理最外侧边框区域
# =====================================================
def remove_outer_frame_strict(binary):
    if binary is None or binary.size == 0:
        return binary

    h, w = binary.shape
    line_mask = np.zeros_like(binary)

    # 1) 上下边框：只在最上/最下区域找长横线
    top_y2 = int(h * 0.24)
    bottom_y1 = int(h * 0.76)

    close_h = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, int(w * 0.025)), 1)
    )
    open_h = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(28, int(w * 0.11)), 1)
    )

    for y1, y2 in [(0, top_y2), (bottom_y1, h)]:
        band = binary[y1:y2, :]
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, close_h)
        band_line = cv2.morphologyEx(band, cv2.MORPH_OPEN, open_h)
        line_mask[y1:y2, :] = cv2.bitwise_or(line_mask[y1:y2, :], band_line)

    # 2) 左右边框：只在极窄外侧找竖线，避免把“沪/苏/浙”等汉字笔画删掉
    side_w = max(6, int(w * 0.045))

    close_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(5, int(h * 0.035)))
    )
    open_v = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(18, int(h * 0.20)))
    )

    for x1, x2 in [(0, side_w), (w - side_w, w)]:
        band = binary[:, x1:x2]
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, close_v)
        band_line = cv2.morphologyEx(band, cv2.MORPH_OPEN, open_v)
        line_mask[:, x1:x2] = cv2.bitwise_or(line_mask[:, x1:x2], band_line)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)

    clean = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))

    # 3) 删除剩下的边缘碎片；条件保守，只删“靠外侧”的小残留
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    remove_mask = np.zeros_like(clean)

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        if area <= 0 or cw <= 0 or ch <= 0:
            continue

        fill_ratio = area / (cw * ch + 1e-6)

        near_left = x < w * 0.025
        near_right = x + cw > w * 0.975
        near_top = y < h * 0.140
        near_bottom = y + ch > h * 0.860

        thin_top_bottom = (
            (near_top or near_bottom)
            and cw > w * 0.080
            and ch < h * 0.140
            and area < h * w * 0.035
        )

        thin_left_right = (
            (near_left or near_right)
            and ch > h * 0.250
            and cw < w * 0.120
            and area < h * w * 0.050
            and fill_ratio < 0.30
        )

        corner_noise = (
            (near_left or near_right)
            and (near_top or near_bottom)
            and area < h * w * 0.055
            and fill_ratio < 0.23
        )

        if thin_top_bottom or thin_left_right or corner_noise:
            remove_mask[labels == i] = 255

    remove_mask = cv2.dilate(remove_mask, kernel, iterations=1)
    clean = cv2.bitwise_and(clean, cv2.bitwise_not(remove_mask))

    return clean


# =====================================================
# 字符槽内删除边框残留：用投影找贴边直线
# =====================================================
def clean_slot_border_by_projection(slot_img, idx):
    if slot_img is None or slot_img.size == 0:
        return slot_img

    clean = slot_img.copy()
    sh, sw = clean.shape

    if sh <= 0 or sw <= 0:
        return clean

    # 不清理每格上/下边缘，因为 text_roi_full 已经按字符高度裁过，
    # 数字 8、9、R 的真实笔画经常正好靠近上边缘。
    # 这里只处理第 1 位最左侧、第 7 位最右侧的外框竖线。
    col_sum = np.sum(clean == 255, axis=0)

    if idx == 0:
        left_limit = max(2, int(sw * 0.14))
        for x in range(0, left_limit):
            if col_sum[x] > sh * 0.42:
                x1 = max(0, x - 2)
                x2 = min(sw, x + 3)
                clean[:, x1:x2] = 0

    if idx == 6:
        right_start = min(sw - 1, int(sw * 0.86))
        for x in range(right_start, sw):
            if col_sum[x] > sh * 0.42:
                x1 = max(0, x - 2)
                x2 = min(sw, x + 3)
                clean[:, x1:x2] = 0

    return clean


# =====================================================
# 车牌二值化：蓝底白字 → 白字黑底
# 输出和模板一致：白字黑底
# =====================================================
def plate_binarization(plate):
    """
    车牌二值化：蓝底白字 → 白字黑底。
    重点修改：
    1. 白字提取不再只依赖 HSV 低饱和高亮度；
       汉字有时偏灰/偏蓝，旧阈值会漏掉。
    2. 使用 HSV + 灰度 + Otsu 组合，保护第一个汉字。
    3. 后续边框和螺丝清理都改成保守版本。
    """
    plate = cv2.resize(plate, (420, 140), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 第一层：HSV 白色/浅色区域
    lower_white = np.array([0, 0, 115])
    upper_white = np.array([180, 150, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    # 第二层：灰度白字
    mask_gray = cv2.inRange(gray, 120, 255)

    # 第三层：Otsu 自动阈值，补上灰白的汉字细笔画
    _, mask_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 蓝底通常饱和度高；白字/边框饱和度低或者亮度高
    mask_sv = np.zeros_like(gray)
    mask_sv[((s < 165) & (v > 110)) | (gray > 145)] = 255

    binary_1 = cv2.bitwise_and(mask_hsv, mask_gray)
    binary_2 = cv2.bitwise_and(mask_otsu, mask_sv)

    binary = cv2.bitwise_or(binary_1, binary_2)

    h, w = binary.shape

    edge_y = max(1, int(h * 0.005))
    edge_x = max(1, int(w * 0.005))

    binary[:edge_y, :] = 0
    binary[h - edge_y:, :] = 0
    binary[:, :edge_x] = 0
    binary[:, w - edge_x:] = 0

    # 只做纵向轻微闭运算，修补断裂，不横向粘连字符
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)

    binary = remove_plate_frame_lines(binary)
    binary = remove_outer_frame_strict(binary)
    binary = remove_screw_noise(binary)
    binary = remove_small_noise_keep_thin_chars(binary)
    binary = remove_outer_frame_strict(binary)

    return binary


# =====================================================
# 字符分割前：温和清理 ROI 左右残余竖边框
# 不再大面积砍左边，避免第一个汉字消失
# =====================================================
def remove_remaining_edge_lines(roi):
    """
    字符分割前清理 ROI 左右残余边框。
    重点修改：
    不再删除“靠近左边”的窄高连通域。
    因为第一个汉字经常就是窄高结构，很容易被误删。
    """
    h, w = roi.shape
    clean = roi.copy()

    # 只清理最外侧 0.6% 的边缘，不碰第一个汉字
    side_cut = max(1, int(w * 0.006))
    clean[:, :side_cut] = 0
    clean[:, w - side_cut:] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        touches_real_left = x <= 1
        touches_real_right = x + cw >= w - 1
        touches_top = y <= 2
        touches_bottom = y + ch >= h - 2

        vertical_edge_line = (
            (touches_real_left or touches_real_right)
            and ch > h * 0.60
            and cw < max(4, w * 0.030)
            and area < h * w * 0.04
        )

        corner_line = (
            (touches_real_left or touches_real_right)
            and (touches_top or touches_bottom)
            and cw > w * 0.08
            and ch < h * 0.12
            and area < h * w * 0.05
        )

        if vertical_edge_line or corner_line:
            clean[labels == i] = 0

    return clean


# =====================================================
# 字符分割前：清理 ROI 上下残余横边框
# 只删除靠近上下边缘的长横线，不删除字符内部笔画
# =====================================================
def remove_horizontal_edge_lines_in_roi(roi):
    """
    字符分割前清理 ROI 上下残余横边框。
    只删除很长的边框线，不删除字符本身的横笔画。
    """
    h, w = roi.shape
    clean = roi.copy()

    line_mask = np.zeros_like(clean)

    top_h = int(h * 0.22)
    bottom_start = int(h * 0.78)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(8, int(w * 0.035)), 1)
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(70, int(w * 0.42)), 1)
    )

    if top_h > 0:
        top_band = clean[:top_h, :]
        top_close = cv2.morphologyEx(top_band, cv2.MORPH_CLOSE, close_kernel)
        top_line = cv2.morphologyEx(top_close, cv2.MORPH_OPEN, open_kernel)
        line_mask[:top_h, :] = top_line

    if bottom_start < h:
        bottom_band = clean[bottom_start:, :]
        bottom_close = cv2.morphologyEx(bottom_band, cv2.MORPH_CLOSE, close_kernel)
        bottom_line = cv2.morphologyEx(bottom_close, cv2.MORPH_OPEN, open_kernel)
        line_mask[bottom_start:, :] = bottom_line

    # 对投影特别长的上下边缘行再补删一次
    row_projection = np.sum(clean == 255, axis=1)

    for y in range(0, top_h):
        if row_projection[y] > w * 0.45:
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            line_mask[y1:y2, :] = cv2.bitwise_or(
                line_mask[y1:y2, :],
                clean[y1:y2, :]
            )

    for y in range(bottom_start, h):
        if row_projection[y] > w * 0.45:
            y1 = max(0, y - 1)
            y2 = min(h, y + 2)
            line_mask[y1:y2, :] = cv2.bitwise_or(
                line_mask[y1:y2, :],
                clean[y1:y2, :]
            )

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

    clean = cv2.bitwise_and(clean, cv2.bitwise_not(line_mask))

    edge_y = max(1, int(h * 0.008))
    clean[:edge_y, :] = 0
    clean[h - edge_y:, :] = 0

    return clean


# =====================================================
# 找字符真正上下范围
# 防止绿色框上下太大，把边框也框进去
# =====================================================
def find_character_vertical_range(roi):
    """
    找字符真正上下范围。
    重点修改：
    如果上下边框线残留，它的行投影会特别长；
    这里忽略靠近上下边缘、投影异常大的行，避免绿色框被边框线撑大。
    """
    h, w = roi.shape

    x1 = int(w * 0.06)
    x2 = int(w * 0.94)

    middle = roi[:, x1:x2]

    row_projection = np.sum(middle == 255, axis=1).astype(np.float32)
    middle_w = middle.shape[1]

    # 上下边缘处投影过长，通常是车牌框线，不让它参与 y 范围计算
    for y in range(h):
        near_edge = y < h * 0.25 or y > h * 0.75

        if near_edge and row_projection[y] > middle_w * 0.38:
            row_projection[y] = 0

    if np.max(row_projection) == 0:
        return int(h * 0.08), int(h * 0.92)

    row_projection = np.convolve(
        row_projection,
        np.ones(3) / 3,
        mode="same"
    )

    threshold = max(2, np.max(row_projection) * 0.10)

    rows = np.where(row_projection > threshold)[0]

    if len(rows) == 0:
        return int(h * 0.08), int(h * 0.92)

    y1 = int(np.min(rows))
    y2 = int(np.max(rows))

    pad = int(h * 0.035)

    y1 = max(0, y1 - pad)
    y2 = min(h - 1, y2 + pad)

    if y2 - y1 > h * 0.88:
        y1 = int(h * 0.06)
        y2 = int(h * 0.94)

    return y1, y2


# =====================================================
# 找车牌字符整体左右范围
# 修复重点：
# 不再用固定 6% / 7% 硬砍左边
# 只过滤极贴边、细长、高度异常的白框
# =====================================================
def find_text_horizontal_range_for_plate(roi):
    h, w = roi.shape

    clean = roi.copy()

    # 只去掉极窄边缘，避免把第一个汉字删掉
    side_cut = max(1, int(w * 0.012))
    clean[:, :side_cut] = 0
    clean[:, w - side_cut:] = 0

    y1 = int(h * 0.16)
    y2 = int(h * 0.90)

    middle = clean[y1:y2, :]
    mid_h = middle.shape[0]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        middle,
        connectivity=8
    )

    valid_x1 = []
    valid_x2 = []

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        if area < 10:
            continue

        extreme_left_frame = (
            x < w * 0.025
            and ch > mid_h * 0.42
            and cw < w * 0.040
        )

        extreme_right_frame = (
            x + cw > w * 0.975
            and ch > mid_h * 0.42
            and cw < w * 0.040
        )

        flat_edge_line = (
            (x < w * 0.025 or x + cw > w * 0.975)
            and ch < mid_h * 0.18
            and cw > w * 0.08
        )

        if extreme_left_frame or extreme_right_frame or flat_edge_line:
            continue

        valid_x1.append(x)
        valid_x2.append(x + cw)

    if len(valid_x1) == 0:
        return int(w * 0.04), int(w * 0.96)

    left = min(valid_x1)
    right = max(valid_x2)

    # 外扩要比上一版更宽松，保护第一个汉字和最后一位
    left = max(int(w * 0.010), left - int(w * 0.035))
    right = min(int(w * 0.990), right + int(w * 0.035))

    if right - left < w * 0.50:
        return int(w * 0.04), int(w * 0.96)

    return left, right


# =====================================================
# 单个字符格内部清理边框残留
# 重点：只删贴边细长边框，不硬砍固定百分比
# =====================================================
def clean_slot_edge_frame(slot_img, idx):
    """
    单个字符格内部清理边框残留。
    重点修改：
    第一位汉字不再删除“靠近左边”的窄高连通域；
    只有真正贴着 slot 最外边界的残余框线才删除。
    """
    clean = slot_img.copy()

    sh, sw = clean.shape

    if sh <= 0 or sw <= 0:
        return clean

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    for li in range(1, num_labels):
        x, y, cw, ch, area = stats[li]

        if area <= 0:
            continue

        touches_top = y <= 1
        touches_bottom = y + ch >= sh - 1

        if idx == 0:
            touches_left = x <= 1

            true_left_frame = (
                touches_left
                and ch > sh * 0.72
                and cw < max(3, sw * 0.080)
                and (touches_top or touches_bottom)
                and area < sh * sw * 0.10
            )

            bottom_left_line = (
                touches_left
                and touches_bottom
                and cw > sw * 0.45
                and ch < sh * 0.12
                and area < sh * sw * 0.10
            )

            if true_left_frame or bottom_left_line:
                clean[labels == li] = 0

        elif idx == 6:
            touches_right = x + cw >= sw - 1

            true_right_frame = (
                touches_right
                and ch > sh * 0.72
                and cw < max(3, sw * 0.080)
                and (touches_top or touches_bottom)
                and area < sh * sw * 0.10
            )

            bottom_right_line = (
                touches_right
                and touches_bottom
                and cw > sw * 0.45
                and ch < sh * 0.12
                and area < sh * sw * 0.10
            )

            if true_right_frame or bottom_right_line:
                clean[labels == li] = 0

    return clean


# =====================================================
# 字符分割：固定蓝牌结构切割
# 返回：
# chars: 7 个字符图像
# roi: 用于显示绿色框的二值图
# draw_regions: 每个绿色框坐标 [x1, y1, x2, y2]
# =====================================================


# =====================================================
# 动态投影找切割线：避免一个字符框吃到后一个字符
# =====================================================
def _find_valley_cut(text_roi, expected_ratio, search_ratio=0.045):
    """
    在 expected_ratio 附近找纵向投影最小的列作为切割线。
    这样比固定比例切割更稳，不容易把后一个字符的白色笔画切进来。
    """
    h, w = text_roi.shape
    if w <= 5:
        return int(w * expected_ratio)

    col_proj = np.sum(text_roi == 255, axis=0).astype(np.float32)
    col_proj = np.convolve(col_proj, np.ones(5, dtype=np.float32) / 5, mode="same")

    center = int(round(w * expected_ratio))
    radius = max(5, int(round(w * search_ratio)))

    left = max(1, center - radius)
    right = min(w - 2, center + radius)

    if right <= left:
        return center

    zone = col_proj[left:right + 1]

    # 优先找近似空白列；没有空白列时找最低的 20% 投影列
    low_threshold = max(1.5, h * 0.045)
    candidates = np.where(zone <= low_threshold)[0]

    if len(candidates) == 0:
        threshold = np.percentile(zone, 20)
        candidates = np.where(zone <= threshold)[0]

    if len(candidates) == 0:
        return center

    # 把候选低谷列合并成连续区间，选择离 expected_ratio 最近的低谷中心
    runs = []
    run_start = candidates[0]
    prev = candidates[0]

    for c in candidates[1:]:
        if c == prev + 1:
            prev = c
        else:
            runs.append((run_start, prev))
            run_start = c
            prev = c

    runs.append((run_start, prev))

    best_cut = center
    best_score = float("inf")

    for a, b in runs:
        mid = left + (a + b) // 2
        run_width = b - a + 1
        local_mean = float(np.mean(col_proj[max(0, mid - 2):min(w, mid + 3)]))

        # 越靠近预期位置、投影越低、低谷越宽越好
        score = abs(mid - center) + 0.25 * local_mean - 0.35 * run_width

        if score < best_score:
            best_score = score
            best_cut = mid

    return int(best_cut)


# =====================================================
# 根据车牌结构 + 投影低谷生成 7 个字符槽
# 汉字 + 字母 + 点 + 5 位字符，其中点单独跳过
# =====================================================
def _get_dynamic_plate_slots(text_roi_full):
    h, w = text_roi_full.shape

    if w <= 20:
        return []

    # 这些是蓝牌字符之间的预期分界点。
    # 不是直接用来硬切，而是在附近找投影低谷。
    expected_cuts = [
        0.145,  # 汉字 | 字母
        0.285,  # 字母 | 点
        0.350,  # 点 | 第3位
        0.495,  # 第3位 | 第4位
        0.620,  # 第4位 | 第5位
        0.745,  # 第5位 | 第6位
        0.870,  # 第6位 | 第7位
    ]

    cuts = []
    for i, r in enumerate(expected_cuts):
        # 点左右两个分界稍微窄搜，避免把 K 或 R 的笔画当成点切掉
        if i in [1, 2]:
            search_ratio = 0.032
        else:
            search_ratio = 0.050
        cuts.append(_find_valley_cut(text_roi_full, r, search_ratio))

    # 保证切割线递增。如果投影搜索失败，就回退到固定比例。
    min_gap = max(4, int(w * 0.030))
    fixed = [int(round(w * r)) for r in expected_cuts]

    valid = True
    prev = 0
    for c in cuts:
        if c <= prev + min_gap:
            valid = False
            break
        prev = c

    if not valid or cuts[-1] >= w - min_gap:
        cuts = fixed

    c1, c2, c3, c4, c5, c6, c7 = cuts

    # 汉字 + 字母 + 跳过点 + 后 5 位
    raw_slots = [
        (0, c1),
        (c1, c2),
        (c3, c4),
        (c4, c5),
        (c5, c6),
        (c6, c7),
        (c7, w),
    ]

    slots = []
    for a, b in raw_slots:
        a = max(0, int(a))
        b = min(w, int(b))
        if b <= a:
            b = min(w, a + 1)
        slots.append((a, b))

    return slots


# =====================================================
# 删除字符槽中来自相邻字符/间隔点的残留
# =====================================================
def _remove_neighbor_residue(slot_img, idx):
    if slot_img is None or slot_img.size == 0:
        return slot_img

    clean = slot_img.copy()
    sh, sw = clean.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean,
        connectivity=8
    )

    for li in range(1, num_labels):
        x, y, cw, ch, area = stats[li]

        if area <= 0 or cw <= 0 or ch <= 0:
            continue

        cx = (x + cw / 2) / sw
        cy = (y + ch / 2) / sh
        fill_ratio = area / (cw * ch + 1e-6)

        touches_left = x <= 1
        touches_right = x + cw >= sw - 1

        # 字母 K 后面的分隔点：小、圆、在右侧，不能进 K 的模板
        dot_after_second_char = (
            idx == 1
            and cx > 0.68
            and 0.20 < cy < 0.80
            and area < sh * sw * 0.075
            and cw < sw * 0.35
            and ch < sh * 0.35
            and fill_ratio > 0.18
        )

        # 第 3 位左边如果吃到点，也删掉
        dot_before_third_char = (
            idx == 2
            and cx < 0.28
            and 0.20 < cy < 0.80
            and area < sh * sw * 0.075
            and cw < sw * 0.35
            and ch < sh * 0.35
            and fill_ratio > 0.18
        )

        # 切割线附近残留：贴边、很窄、面积小，通常是后一个字符被吃进来的连接块
        right_edge_residue = (
            idx < 6
            and touches_right
            and cw < sw * 0.22
            and area < sh * sw * 0.16
        )

        left_edge_residue = (
            idx > 0
            and touches_left
            and cw < sw * 0.22
            and area < sh * sw * 0.16
        )

        # 最后一位右侧还可能带车牌外框残留，也单独清掉
        right_outer_frame_residue = (
            idx == 6
            and x > sw * 0.70
            and cw < sw * 0.16
            and area < sh * sw * 0.18
        )

        # 但 I / 1 这种字符可能本身很窄，所以不要对整格唯一大连通域乱删。
        # 只有在该残留不是本格主要部分时才删。
        total_white = max(1, np.sum(clean == 255))
        is_minor_component = area / total_white < 0.35

        if dot_after_second_char or dot_before_third_char:
            clean[labels == li] = 0
        elif is_minor_component and (
            right_edge_residue or left_edge_residue or right_outer_frame_residue
        ):
            clean[labels == li] = 0

    return clean


# =====================================================
# 对单个字符槽做最终裁剪：不再向隔壁字符方向扩张
# =====================================================
def _crop_single_character_from_slot(slot_img, idx):
    if slot_img is None or slot_img.size == 0:
        return slot_img, (0, 0, max(0, slot_img.shape[1] - 1), max(0, slot_img.shape[0] - 1))

    clean = slot_img.copy()

    clean = clean_slot_edge_frame(clean, idx)
    clean = clean_slot_border_by_projection(clean, idx)
    clean = remove_screw_noise_in_slot(clean, idx)
    clean = _remove_neighbor_residue(clean, idx)

    ys, xs = np.where(clean == 255)

    if len(xs) == 0 or len(ys) == 0:
        h, w = clean.shape
        return clean, (0, 0, w - 1, h - 1)

    sx1 = int(np.min(xs))
    sx2 = int(np.max(xs))
    sy1 = int(np.min(ys))
    sy2 = int(np.max(ys))

    sh, sw = clean.shape

    # 只给很小的安全边距；不再像旧版那样 pad 8%，否则会把后一个字符连进来
    pad_x = 1
    pad_y = max(1, int(sh * 0.025))

    sx1 = max(0, sx1 - pad_x)
    sx2 = min(sw - 1, sx2 + pad_x)
    sy1 = max(0, sy1 - pad_y)
    sy2 = min(sh - 1, sy2 + pad_y)

    char_img = clean[sy1:sy2 + 1, sx1:sx2 + 1]

    return char_img, (sx1, sy1, sx2, sy2)


def segment_characters(binary):
    h, w = binary.shape

    roi = binary[int(h * 0.04):int(h * 0.96), int(w * 0.02):int(w * 0.98)]

    roi = remove_remaining_edge_lines(roi)
    roi = remove_horizontal_edge_lines_in_roi(roi)
    roi = remove_outer_frame_strict(roi)

    rh, rw = roi.shape
    side_cut = max(1, int(rw * 0.010))
    roi[:, :side_cut] = 0
    roi[:, rw - side_cut:] = 0

    y1, y2 = find_character_vertical_range(roi)
    x_left, x_right = find_text_horizontal_range_for_plate(roi)

    text_roi_full = roi[y1:y2 + 1, x_left:x_right]

    th, tw = text_roi_full.shape

    if tw <= 0 or th <= 0:
        return [], roi, []

    # 新版核心：不用固定比例硬切；先在预期位置附近找投影低谷，
    # 再跳过中间的圆点，避免 K 吃到点、R 吃到 9。
    slots = _get_dynamic_plate_slots(text_roi_full)

    if len(slots) != 7:
        return [], roi, []

    chars = []
    draw_regions = []

    for idx, (a, b) in enumerate(slots):
        slot_img = text_roi_full[:, a:b].copy()

        char_img, (sx1, sy1, sx2, sy2) = _crop_single_character_from_slot(
            slot_img,
            idx
        )

        char_norm = resize_and_center(char_img)
        chars.append(char_norm)

        draw_x1 = x_left + a + sx1
        draw_x2 = x_left + a + sx2
        draw_y1 = y1 + sy1
        draw_y2 = y1 + sy2

        draw_regions.append([
            int(draw_x1),
            int(draw_y1),
            int(draw_x2),
            int(draw_y2)
        ])

    return chars, roi, draw_regions


# =====================================================
# 特征提取：像素 + 垂直投影 + 水平投影
# =====================================================
def extract_feature(char_img):
    img = char_img.astype(np.float32) / 255.0

    pixel_feature = img.flatten()

    vertical_projection = np.sum(img, axis=0)
    horizontal_projection = np.sum(img, axis=1)

    if np.max(vertical_projection) > 0:
        vertical_projection = vertical_projection / np.max(vertical_projection)

    if np.max(horizontal_projection) > 0:
        horizontal_projection = horizontal_projection / np.max(horizontal_projection)

    feature = np.concatenate([
        pixel_feature,
        vertical_projection * 2.0,
        horizontal_projection * 2.0
    ])

    return feature


# =====================================================
# 平移字符图，减少切割偏移误差
# =====================================================
def shift_image(img, dx, dy):
    h, w = img.shape

    matrix = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    shifted = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return shifted


# =====================================================
# 欧氏距离模板匹配
# =====================================================
def match_template_euclidean(char_img, template_dict):
    if char_img is None or np.sum(char_img == 255) < 8:
        return "?", float("inf")

    char_features = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shifted = shift_image(char_img, dx, dy)
            char_features.append(extract_feature(shifted))

    best_label = ""
    best_distance = float("inf")

    for label, template in template_dict.items():
        template_feature = extract_feature(template)

        for feature in char_features:
            distance = np.sqrt(np.sum((feature - template_feature) ** 2))

            if distance < best_distance:
                best_distance = distance
                best_label = label

    return best_label, best_distance


# =====================================================
# 统计字符内部空洞数量
# 用于修正 8/0/Q、9/2、B/R
# =====================================================
def count_inner_holes(char_img):
    if char_img is None or char_img.size == 0:
        return 0

    _, binary = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape

    black = (binary == 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        black,
        connectivity=8
    )

    holes = 0

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]

        touches_border = (
            x == 0 or y == 0 or x + cw >= w or y + ch >= h
        )

        if touches_border:
            continue

        if area < 8:
            continue

        holes += 1

    return holes


# =====================================================
# 字符结构特征
# =====================================================
def get_structural_features(char_img):
    _, binary = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

    h, w = binary.shape
    total = np.sum(binary == 255) + 1

    holes = count_inner_holes(binary)

    left_part = binary[:, :int(w * 0.25)]
    bottom_part = binary[int(h * 0.75):, :]
    lower_right = binary[int(h * 0.58):, int(w * 0.55):]
    lower_left = binary[int(h * 0.58):, :int(w * 0.45)]

    left_stem = np.sum(left_part == 255) / total
    bottom_bar = np.sum(bottom_part == 255) / total
    lower_right_ratio = np.sum(lower_right == 255) / total
    lower_left_ratio = np.sum(lower_left == 255) / total

    tail_score = lower_right_ratio - lower_left_ratio

    return {
        "holes": holes,
        "left_stem": left_stem,
        "bottom_bar": bottom_bar,
        "tail_score": tail_score,
        "lower_right_ratio": lower_right_ratio
    }


# =====================================================
# 识别后结构纠错
# 作用：模板匹配容易把 T/Y、1/Y、5/Q、R/Q、B/G、K/Q、D/Q、S/0、冀/赣 混淆。
# 这里不替代模板匹配，只在容易混淆的字符上用形状特征做二次修正。
# =====================================================
def get_shape_metrics(char_img):
    if char_img is None or char_img.size == 0:
        return None

    if len(char_img.shape) == 3:
        gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = char_img.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    white = (binary == 255).astype(np.uint8)

    ys, xs = np.where(white == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(np.min(xs)), int(np.max(xs))
    y1, y2 = int(np.min(ys)), int(np.max(ys))

    crop = white[y1:y2 + 1, x1:x2 + 1]
    h, w = crop.shape

    if h <= 0 or w <= 0:
        return None

    row = np.sum(crop, axis=1).astype(np.float32)
    col = np.sum(crop, axis=0).astype(np.float32)

    def band_density(y_a, y_b, x_a=0, x_b=None):
        if x_b is None:
            x_b = w
        y_a = max(0, min(h, int(y_a)))
        y_b = max(0, min(h, int(y_b)))
        x_a = max(0, min(w, int(x_a)))
        x_b = max(0, min(w, int(x_b)))
        if y_b <= y_a or x_b <= x_a:
            return 0.0
        band = crop[y_a:y_b, x_a:x_b]
        return float(np.sum(band)) / float(band.size + 1e-6)

    top_end = max(1, int(h * 0.22))
    mid_a = max(0, int(h * 0.36))
    mid_b = min(h, int(h * 0.66))
    bot_a = max(0, int(h * 0.76))

    left_end = max(1, int(w * 0.28))
    center_a = max(0, int(w * 0.35))
    center_b = min(w, max(center_a + 1, int(w * 0.65)))
    right_a = max(0, int(w * 0.72))

    features = get_structural_features(binary)

    return {
        "bbox_w": w,
        "bbox_h": h,
        "aspect": float(w) / float(h + 1e-6),
        "density": float(np.sum(crop)) / float(w * h + 1e-6),
        "holes": int(features["holes"]),
        "left_stem": float(features["left_stem"]),
        "bottom_bar": float(features["bottom_bar"]),
        "tail_score": float(features["tail_score"]),
        "lower_right_ratio": float(features["lower_right_ratio"]),
        "top_density": band_density(0, top_end),
        "mid_density": band_density(mid_a, mid_b),
        "bottom_density": band_density(bot_a, h),
        "left_density": band_density(0, h, 0, left_end),
        "center_density": band_density(0, h, center_a, center_b),
        "right_density": band_density(0, h, right_a, w),
        "top_max_row": float(np.max(row[:top_end])) / float(w),
        "mid_max_row": float(np.max(row[mid_a:mid_b])) / float(w) if mid_b > mid_a else 0.0,
        "bottom_max_row": float(np.max(row[bot_a:])) / float(w),
        "max_col": float(np.max(col)) / float(h),
        "center_max_col": float(np.max(col[center_a:center_b])) / float(h) if center_b > center_a else 0.0,
    }


def looks_like_I(m):
    if m is None:
        return False

    # I 和数字 1 都是细长竖线。
    # I 的区别通常是上下有更明显的横向帽子/端点，
    # 所以不能只因为“窄 + 高 + 竖直”就把它改成 1。
    has_top_cap = m["top_max_row"] >= 0.44
    has_bottom_cap = m["bottom_max_row"] >= 0.44

    return (
        m["holes"] == 0
        and m["bbox_h"] >= 34
        and m["aspect"] <= 0.38
        and m["max_col"] >= 0.72
        and m["center_max_col"] >= 0.58
        and (has_top_cap or has_bottom_cap)
    )


def looks_like_one(m):
    if m is None:
        return False

    # 数字 1：细长竖线，但不能把带明显 I 形上下端点的字符也归成 1。
    return (
        m["holes"] == 0
        and m["bbox_h"] >= 34
        and m["aspect"] <= 0.30
        and m["max_col"] >= 0.82
        and not looks_like_I(m)
    )


def looks_like_T(m):
    if m is None:
        return False
    return (
        m["holes"] == 0
        and m["top_max_row"] >= 0.82
        and m["center_max_col"] >= 0.78
        and m["left_density"] <= 0.35
        and m["right_density"] <= 0.42
        and m["bottom_density"] <= 0.46
    )


def looks_like_E(m):
    if m is None:
        return False
    return (
        m["holes"] == 0
        and m["left_stem"] >= 0.28
        and m["top_max_row"] >= 0.78
        and m["mid_max_row"] >= 0.65
        and m["bottom_max_row"] >= 0.78
        and m["left_density"] >= 0.55
    )


def looks_like_D(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and m["left_stem"] >= 0.30
        and m["left_density"] >= 0.82
        and m["right_density"] >= 0.70
        and m["center_density"] <= 0.48
        and abs(m["tail_score"]) <= 0.045
        and m["top_max_row"] >= 0.78
        and m["bottom_max_row"] >= 0.78
    )


def looks_like_R(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and m["left_stem"] >= 0.30
        and m["bottom_bar"] <= 0.17
        and m["tail_score"] <= 0.045
        and m["left_density"] >= 0.70
    )


def looks_like_B(m):
    if m is None:
        return False

    # B 的核心特征：左侧竖干很强，通常有上下两个孔；
    # 有时二值化/螺丝会让两个孔粘成一个，所以这里给 holes >= 1 留余量。
    return (
        m["holes"] >= 1
        and m["left_stem"] >= 0.28
        and m["left_density"] >= 0.70
        and m["top_max_row"] >= 0.65
        and m["mid_max_row"] >= 0.55
        and m["bottom_max_row"] >= 0.65
        and m["tail_score"] <= 0.075
    )


def looks_like_G(m):
    if m is None:
        return False

    # G 通常只有一个孔/内凹，左侧竖干没有 B 那么满，
    # 右下会有开口和短横，不应该有 B 那种强左竖干 + 双孔结构。
    return (
        m["holes"] <= 1
        and m["left_stem"] < 0.28
        and m["left_density"] < 0.72
        and m["top_max_row"] >= 0.55
        and m["bottom_max_row"] >= 0.50
    )


def looks_like_K(m):
    if m is None:
        return False
    return (
        m["holes"] == 0
        and m["left_stem"] >= 0.29
        and m["left_density"] >= 0.72
        and m["mid_max_row"] >= 0.65
        and m["aspect"] <= 0.62
    )


def looks_like_5(m):
    if m is None:
        return False
    return (
        m["holes"] == 0
        and m["top_max_row"] >= 0.82
        and m["bottom_max_row"] >= 0.75
        and m["left_stem"] >= 0.22
        and m["bottom_bar"] <= 0.18
        and m["left_density"] >= 0.55
    )


def looks_like_S(m):
    if m is None:
        return False
    return (
        m["holes"] == 0
        and m["top_max_row"] >= 0.75
        and m["bottom_max_row"] >= 0.75
        and 0.13 <= m["left_stem"] <= 0.24
        and m["bottom_bar"] >= 0.17
        and m["tail_score"] >= 0.055
    )


def looks_like_9(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and m["left_stem"] <= 0.19
        and m["bottom_bar"] <= 0.12
        and m["bottom_max_row"] <= 0.48
        and m["top_max_row"] >= 0.72
    )


def looks_like_6(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and m["top_max_row"] <= 0.55
        and m["bottom_max_row"] >= 0.65
        and m["bottom_bar"] >= 0.15
        and m["left_stem"] <= 0.24
        and abs(m["tail_score"]) <= 0.050
    )


def looks_like_Q(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and m["tail_score"] >= 0.060
        and m["lower_right_ratio"] >= 0.21
        and m["bottom_bar"] >= 0.15
    )


def looks_like_zero(m):
    if m is None:
        return False
    return (
        m["holes"] == 1
        and 0.19 <= m["left_stem"] <= 0.31
        and abs(m["tail_score"]) <= 0.055
        and m["bottom_bar"] >= 0.13
        and m["center_density"] <= 0.50
    )


def post_correct_second_letter(label, char_img):
    label = label.upper()
    m = get_shape_metrics(char_img)

    # 第二位一定是英文字母；如果模板识别成 I，不能再按数字 1 的逻辑处理。
    if label == "I" or looks_like_I(m):
        return "I"

    # 第二位一定是英文字母，所以这里可以比后 5 位更大胆一点。
    # 重点修复：B 容易被模板匹配成 G。
    if label == "G" and looks_like_B(m):
        return "B"
    if label == "B" and looks_like_G(m):
        return "G"
    if looks_like_B(m) and label in ["B", "G", "R", "8"]:
        return "B"

    if looks_like_T(m):
        return "T"
    if looks_like_E(m):
        return "E"
    if looks_like_D(m):
        return "D"
    if looks_like_K(m):
        return "K"
    if looks_like_S(m):
        return "S"
    if looks_like_R(m):
        return "R"

    return label


def post_correct_alnum(label, char_img, idx):
    label = label.upper()
    m = get_shape_metrics(char_img)

    # 后五位允许字母和数字。这里重点修复 B/G/R/8 的混淆。
    if label == "G" and looks_like_B(m):
        return "B"
    if label == "B" and looks_like_G(m):
        return "G"
    if looks_like_B(m) and label in ["B", "G", "R", "8"]:
        return "B"

    # 这些规则只处理最常见误判，不强行改所有字符。
    # 注意：如果模板已经判断成 I，就不要再因为它是窄竖线而强制改成 1。
    if label == "I":
        return "I"

    if looks_like_I(m) and label in ["1", "I", "T", "Y"]:
        return "I"

    if looks_like_one(m):
        return "1"

    if looks_like_5(m):
        return "5"

    if looks_like_S(m):
        return "S"

    if looks_like_K(m):
        return "K"

    if looks_like_R(m):
        return "R"

    if looks_like_9(m):
        return "9"

    if looks_like_6(m):
        return "6"

    if looks_like_Q(m):
        return "Q"

    if looks_like_zero(m):
        return "0"

    return label


# =====================================================
# 汉字易混纠错
# 重点修复：冀 被识别成 赣。
# 注意：这里不是无条件把 赣 改成 冀，而是只有在两者模板距离接近时才纠正。
# 如果你的真实数据里有很多“赣”牌，并且被误改成“冀”，把 1.28 改小到 1.12。
# =====================================================
def distance_to_chinese_label(char_img, templates, target_label):
    template_dict = templates.get("chinese", {})

    real_label, template = get_template_by_label(template_dict, target_label)

    if template is None:
        return float("inf")

    template_feature = extract_feature(template)

    best_distance = float("inf")

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shifted = shift_image(char_img, dx, dy)
            char_feature = extract_feature(shifted)
            distance = np.sqrt(np.sum((char_feature - template_feature) ** 2))

            if distance < best_distance:
                best_distance = distance

    return best_distance


def post_correct_chinese(label, char_img, templates):
    # 河北“冀”和江西“赣”都是复杂汉字，模板匹配很容易被螺丝/切割噪声带偏。
    if label not in ["冀", "赣"]:
        return label

    d_ji = distance_to_chinese_label(char_img, templates, "冀")
    d_gan = distance_to_chinese_label(char_img, templates, "赣")

    if d_ji == float("inf") or d_gan == float("inf"):
        return label

    # 如果已经明显更像某一个，就尊重模板距离。
    if d_ji < d_gan * 0.90:
        return "冀"
    if d_gan < d_ji * 0.78:
        return "赣"

    # 关键修复：当系统输出“赣”，但“冀”的距离只是略大，通常是冀的笔画被噪声打散了。
    # 这时优先纠成“冀”。
    if label == "赣" and d_ji <= d_gan * 1.28:
        return "冀"

    return label


# =====================================================
# 根据标签找到模板
# =====================================================
def get_template_by_label(template_dict, target_label):
    target_label = target_label.upper()

    for label, template in template_dict.items():
        if label.upper() == target_label:
            return label.upper(), template

    return None, None


# =====================================================
# 与指定模板计算距离
# =====================================================
def distance_to_specific_label(char_img, templates, target_label):
    target_label = target_label.upper()

    if target_label.isdigit():
        template_dict = templates["number"]
    else:
        template_dict = templates["letter"]

    real_label, template = get_template_by_label(template_dict, target_label)

    if template is None:
        return float("inf")

    template_feature = extract_feature(template)

    best_distance = float("inf")

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            shifted = shift_image(char_img, dx, dy)
            char_feature = extract_feature(shifted)

            distance = np.sqrt(np.sum((char_feature - template_feature) ** 2))

            if distance < best_distance:
                best_distance = distance

    return best_distance


# =====================================================
# 易混字符结构惩罚
# =====================================================
def structural_penalty(label, features):
    label = label.upper()

    holes = features["holes"]
    tail = features["tail_score"]
    bottom_bar = features["bottom_bar"]
    left_stem = features["left_stem"]
    lower_right = features["lower_right_ratio"]

    penalty = 0.0

    # 8 / 0 / Q
    if label == "8":
        if holes >= 2:
            penalty -= 0.20
        else:
            penalty += 0.45

        if tail > 0.08:
            penalty += 0.20

    elif label == "0":
        if holes == 1:
            penalty -= 0.10
        else:
            penalty += 0.40

        if tail > 0.08:
            penalty += 0.25

    elif label == "Q":
        if holes == 1:
            penalty -= 0.10
        else:
            penalty += 0.40

        if tail < 0.04 and lower_right < 0.18:
            penalty += 0.30
        else:
            penalty -= 0.10

    # 9 / 2
    elif label == "9":
        if holes >= 1:
            penalty -= 0.15
        else:
            penalty += 0.35

        if bottom_bar > 0.28 and holes == 0:
            penalty += 0.25

    elif label == "2":
        if holes == 0:
            penalty -= 0.10
        else:
            penalty += 0.45

        if bottom_bar < 0.18:
            penalty += 0.15

    # B / G / R
    elif label == "B":
        if holes >= 2:
            penalty -= 0.35
        elif holes == 1 and left_stem >= 0.28:
            penalty -= 0.12
        else:
            penalty += 0.45

        if left_stem < 0.24:
            penalty += 0.22

        if tail > 0.09:
            penalty += 0.20

    elif label == "G":
        if holes <= 1:
            penalty -= 0.05
        else:
            penalty += 0.38

        # B 的左竖干很强；如果左竖干很强，就不应该偏向 G。
        if left_stem >= 0.28:
            penalty += 0.30

        if tail > 0.10:
            penalty -= 0.05

    elif label == "R":
        if holes <= 1:
            penalty -= 0.10
        else:
            penalty += 0.30

        if tail < 0.04 and lower_right < 0.18:
            penalty += 0.25
        else:
            penalty -= 0.10

    return penalty


# =====================================================
# 从易混字符组中重新选择
# =====================================================
def choose_from_ambiguous_group(char_img, templates, candidate_labels):
    features = get_structural_features(char_img)

    distances = {}

    for label in candidate_labels:
        distances[label] = distance_to_specific_label(char_img, templates, label)

    finite_distances = [
        d for d in distances.values()
        if d != float("inf")
    ]

    if len(finite_distances) == 0:
        return candidate_labels[0], float("inf")

    min_distance = min(finite_distances)

    best_label = candidate_labels[0]
    best_score = float("inf")

    for label in candidate_labels:
        distance = distances[label]

        if distance == float("inf"):
            continue

        distance_score = distance / (min_distance + 1e-6)
        penalty = structural_penalty(label, features)

        final_score = distance_score + penalty

        if final_score < best_score:
            best_score = final_score
            best_label = label

    return best_label, best_score


# =====================================================
# 字母数字混合识别
# =====================================================
def match_alnum(char_img, templates):
    label_l, dist_l = match_template_euclidean(char_img, templates["letter"])
    label_n, dist_n = match_template_euclidean(char_img, templates["number"])

    label_l = label_l.upper()

    # I / 1 是最容易混的一组。
    # 这里先单独处理，避免后面的结构纠错把 I 强行改成 1。
    # 当字母模板 I 的距离和数字模板 1 很接近时，优先保留 I。
    # 如果你的真实数字 1 被误改成 I，可以把 1.18 调小到 1.08。
    if label_l == "I" and label_n == "1":
        if dist_l <= dist_n * 1.18:
            return "I", dist_l
        else:
            return "1", dist_n

    if dist_l < dist_n:
        raw_label = label_l
        raw_dist = dist_l
    else:
        raw_label = label_n
        raw_dist = dist_n

    if raw_label in ["9", "2"] or label_n in ["9", "2"]:
        return choose_from_ambiguous_group(
            char_img,
            templates,
            ["9", "2"]
        )

    if raw_label in ["8", "0", "Q"] or label_n in ["8", "0"] or label_l in ["Q"]:
        return choose_from_ambiguous_group(
            char_img,
            templates,
            ["8", "0", "Q"]
        )

    if raw_label in ["B", "G", "R"] or label_l in ["B", "G", "R"]:
        return choose_from_ambiguous_group(
            char_img,
            templates,
            ["B", "G", "R"]
        )

    return raw_label, raw_dist


# =====================================================
# 按车牌规则识别
# 第1位：汉字
# 第2位：字母
# 后5位：字母或数字
# =====================================================
def recognize_plate(chars, templates):
    if len(chars) != 7:
        return f"字符分割失败：{len(chars)}"

    result = ""

    for i, char_img in enumerate(chars):
        if i == 0:
            label, dist = match_template_euclidean(
                char_img,
                templates["chinese"]
            )
            label = post_correct_chinese(label, char_img, templates)

        elif i == 1:
            label, dist = match_template_euclidean(
                char_img,
                templates["letter"]
            )

            label = label.upper()

            if label in ["B", "G", "R"]:
                label, dist = choose_from_ambiguous_group(
                    char_img,
                    templates,
                    ["B", "G", "R"]
                )

            label = post_correct_second_letter(label, char_img)

        else:
            label, dist = match_alnum(char_img, templates)
            label = post_correct_alnum(label, char_img, i)

        result += label

    return result


# =====================================================
# 显示处理过程
# =====================================================
def show_process(image, blue_mask, box, plate, binary, roi, regions, chars):
    show_img = image.copy()

    x1, y1, x2, y2 = box

    cv2.rectangle(
        show_img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        3
    )

    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    for region in regions:
        rx1, ry1, rx2, ry2 = region

        cv2.rectangle(
            roi_color,
            (int(rx1), int(ry1)),
            (int(rx2), int(ry2)),
            (0, 255, 0),
            1
        )

    cv2.imshow("1 Original Image", resize_show(image))
    cv2.imshow("2 Blue Mask", resize_show(blue_mask))
    cv2.imshow("3 Plate Location", resize_show(show_img))
    cv2.imshow("4 Cropped Plate", resize_show(plate))
    cv2.imshow("5 Binary Plate - White Text Black Background", resize_show(binary))
    cv2.imshow("6 Character Segmentation - Green Boxes", resize_show(roi_color))

    for i, ch in enumerate(chars):
        cv2.imshow(f"char_{i + 1}", ch)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# 单张图片识别完整流程
# =====================================================
def recognize_one_image(image_path, templates):
    image = cv_imread(image_path)

    if image is None:
        return "图片读取失败"

    plate, blue_mask, box = locate_blue_plate(image)

    if plate is None:
        return "未检测到车牌"

    binary = plate_binarization(plate)

    chars, roi, regions = segment_characters(binary)

    result = recognize_plate(chars, templates)

    if SHOW_PROCESS:
        show_process(image, blue_mask, box, plate, binary, roi, regions, chars)

    return result


# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":
    templates = load_templates(TEMPLATE_ROOT)

    print("-" * 45)
    print("开始识别车牌")
    print("模板类型：白字黑底")
    print("切割字符类型：白字黑底")
    print("-" * 45)

    for image_path in IMAGE_PATHS:
        result = recognize_one_image(image_path, templates)
        print(image_path, "识别结果：", result)

    print("-" * 45)
    print("识别完成")