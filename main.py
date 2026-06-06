# main.py

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os

# ─────────────────────────────────────────
# Инициализация моделей
# ─────────────────────────────────────────

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

yolo_model = YOLO("yolo11n-seg.pt")

# Загрузка картинок для Framemom Meter (положи файлы рядом с main.py):
# framemom_klavikular.png, framemom_david.png, framemom_chud.png
script_dir = os.path.dirname(__file__)
img_klav = cv2.imread(os.path.join(script_dir, "framemom_klavikular.png"), cv2.IMREAD_UNCHANGED)
img_david = cv2.imread(os.path.join(script_dir, "framemom_david.png"), cv2.IMREAD_UNCHANGED)
img_chud = cv2.imread(os.path.join(script_dir, "framemom_chud.png"), cv2.IMREAD_UNCHANGED)
# Проверяем, успешно ли загружены изображения (None означает, что файл не найден или не может быть прочитан)
if img_klav is None:
    print(f"Warning: framemom_klavikular.png not found or unreadable in {script_dir}")
if img_david is None:
    print(f"Warning: framemom_david.png not found or unreadable in {script_dir}")
if img_chud is None:
    print(f"Warning: framemom_chud.png not found or unreadable in {script_dir}")

def overlay_image(bg, fg, x, y, scale=1.0):
    """Overlay `fg` onto `bg` at position (x,y). Supports FG with alpha channel."""
    if fg is None:
        return bg
    h_fg, w_fg = fg.shape[:2]
    w_new = int(w_fg * scale)
    h_new = int(h_fg * scale)
    if w_new <= 0 or h_new <= 0:
        return bg
    fg_resized = cv2.resize(fg, (w_new, h_new), interpolation=cv2.INTER_AREA)

    # region on background
    h_bg, w_bg = bg.shape[:2]
    if x >= w_bg or y >= h_bg:
        return bg

    x_end = min(w_bg, x + w_new)
    y_end = min(h_bg, y + h_new)
    fg_region = fg_resized[0:y_end - y, 0:x_end - x]

    if fg_region.shape[2] == 4:
        alpha = fg_region[:, :, 3] / 255.0
        for c in range(3):
            bg[y:y_end, x:x_end, c] = (alpha * fg_region[:, :, c] + (1 - alpha) * bg[y:y_end, x:x_end, c])
    else:
        # simple blend
        bg[y:y_end, x:x_end] = fg_region

    return bg

# ─────────────────────────────────────────
# Вспомогательная функция
# ─────────────────────────────────────────

def find_contour_x_at_y(mask_binary, target_y, side):
    """
    Находит крайний X пиксель маски на заданном Y.
    
    mask_binary : бинарная маска (0/255), размер = кадру
    target_y    : Y-координата в пикселях (от MediaPipe)
    side        : 'left' → самый левый X, 'right' → самый правый X
    
    Возвращает X или None если строка пустая.
    Сканируем не одну строку а диапазон ±SCAN_RANGE пикселей —
    чтобы не промахнуться если маска на этом Y имеет дырку.
    """
    SCAN_RANGE = 10  # пикселей вверх/вниз от target_y

    h, w = mask_binary.shape
    y_min = max(0, target_y - SCAN_RANGE)
    y_max = min(h - 1, target_y + SCAN_RANGE)

    # Берём полосу строк и схлопываем в одну через OR
    strip = mask_binary[y_min:y_max + 1, :]  # shape: (SCAN_RANGE*2, w)
    row = np.any(strip > 0, axis=0)          # shape: (w,) — True где тело есть

    indices = np.where(row)[0]  # все X где маска = 1

    if len(indices) == 0:
        return None

    if side == 'left':
        return int(indices[0])   # самый левый X
    else:
        return int(indices[-1])  # самый правый X


def find_contour_x_near_y(mask_binary, target_y, target_x, side):
    """
    Находит X контура поблизости от целевой точки (target_x, target_y).
    Используется для поиска контура в окрестности MediaPipe ориентира,
    чтобы избежать путаницы с другими частями тела (руки, голова).
    
    mask_binary : бинарная маска (0/255), размер = кадру
    target_y    : Y-координата в пикселях (от MediaPipe)
    target_x    : X-координата в пикселях (от MediaPipe) — примерный центр поиска
    side        : 'left' → ищем край влево от target_x
                   'right' → ищем край вправо от target_x
    
    Возвращает X или None если контур не найден.
    """
    SCAN_RANGE_Y = 10  # пикселей вверх/вниз от target_y
    SEARCH_RANGE = 200  # пикселей влево/вправо от target_x для поиска

    h, w = mask_binary.shape
    y_min = max(0, target_y - SCAN_RANGE_Y)
    y_max = min(h - 1, target_y + SCAN_RANGE_Y)

    # Берём полосу строк
    strip = mask_binary[y_min:y_max + 1, :]
    row = np.any(strip > 0, axis=0)  # True где маска = 1

    indices = np.where(row)[0]  # все X где есть контур

    if len(indices) == 0:
        return None

    if side == 'left':
        # Ищем крайнее слева, но в диапазоне от (target_x - SEARCH_RANGE) до target_x
        x_min = max(0, target_x - SEARCH_RANGE)
        valid_indices = indices[(indices >= x_min) & (indices <= target_x)]
        if len(valid_indices) > 0:
            return int(valid_indices[0])  # самый левый в этом диапазоне
        return None
    else:  # 'right'
        # Ищем крайнее справа, но в диапазоне от target_x до (target_x + SEARCH_RANGE)
        x_max = min(w - 1, target_x + SEARCH_RANGE)
        valid_indices = indices[(indices >= target_x) & (indices <= x_max)]
        if len(valid_indices) > 0:
            return int(valid_indices[-1])  # самый правый в этом диапазоне
        return None


def find_contour_x_in_range(mask_binary, target_y, x_min, x_max, side):
    """
    Находит X контура внутри заданного диапазона [x_min, x_max].
    Используется для поиска контура туловища между известными границами.
    
    mask_binary : бинарная маска (0/255), размер = кадру
    target_y    : Y-координата в пикселях
    x_min, x_max: диапазон поиска по X
    side        : 'left' → самый левый X в диапазоне
                   'right' → самый правый X в диапазоне
    
    Возвращает X или None если контур не найден.
    """
    SCAN_RANGE_Y = 10  # пикселей вверх/вниз от target_y

    h, w = mask_binary.shape
    y_min = max(0, target_y - SCAN_RANGE_Y)
    y_max = min(h - 1, target_y + SCAN_RANGE_Y)

    # Берём полосу строк
    strip = mask_binary[y_min:y_max + 1, :]
    row = np.any(strip > 0, axis=0)  # True где маска = 1

    indices = np.where(row)[0]  # все X где есть контур

    if len(indices) == 0:
        return None

    # Фильтруем только по заданному диапазону
    valid_indices = indices[(indices >= x_min) & (indices <= x_max)]
    
    if len(valid_indices) == 0:
        return None

    if side == 'left':
        return int(valid_indices[0])   # самый левый в диапазоне
    else:  # 'right'
        return int(valid_indices[-1])  # самый правый в диапазоне

# ─────────────────────────────────────────
# Камера
# ─────────────────────────────────────────

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Камера не найдена")
    exit()

print("Камера запущена. Нажми Q для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Общая маска тела для этого кадра (заполняется ниже)
    body_mask = np.zeros((h, w), dtype=np.uint8)

    # ─────────────────────────────────────
    # YOLO: сегментация — строим маску и рисуем контур
    # ─────────────────────────────────────

    yolo_results = yolo_model(frame, verbose=False)

    for result in yolo_results:
        if result.masks is None:
            continue

        for i, mask in enumerate(result.masks.data):
            cls = int(result.boxes.cls[i].item())
            if cls != 0:  # только person
                continue

            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            # Накапливаем маску (если несколько людей — OR)
            body_mask = cv2.bitwise_or(body_mask, mask_binary)

            # Контур тела — зелёный
            contours, _ = cv2.findContours(
                mask_binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            # Полупрозрачная заливка
            overlay = frame.copy()
            cv2.fillPoly(overlay, contours, (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)

    # ─────────────────────────────────────
    # MediaPipe: landmark-точки
    # ─────────────────────────────────────

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = pose.process(rgb_frame)

    if mp_results.pose_landmarks:
        landmarks = mp_results.pose_landmarks.landmark

        # Скелет рисовать не будем — убираем точки лица/мелкие маркеры
        # (мы всё ещё рисуем отдельные желтые маркеры для плеч/бедер)

        # Исходные точки MediaPipe (жёлтые — для сравнения)
        ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        rh = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        ls_px = (int(ls.x * w), int(ls.y * h))
        rs_px = (int(rs.x * w), int(rs.y * h))
        lh_px = (int(lh.x * w), int(lh.y * h))
        rh_px = (int(rh.x * w), int(rh.y * h))

        # Жёлтые точки — оригинал MediaPipe (для сравнения)
        for pt, label in [(ls_px, "MP_L"), (rs_px, "MP_R")]:
            cv2.circle(frame, pt, 6, (0, 255, 255), -1)
            cv2.putText(frame, label, (pt[0] + 8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ─────────────────────────────────
        # НОВЫЕ ТОЧКИ: X из контура, Y из MediaPipe
        # ─────────────────────────────────

        # Y плеч от MediaPipe (в пикселях) — используем среднее двух плеч
        # чтобы сканировать на одном горизонтальном уровне
        shoulder_y = int((ls.y + rs.y) / 2 * h)

        # Ищем крайние X маски на уровне плеч
        left_x  = find_contour_x_at_y(body_mask, shoulder_y, side='left')
        right_x = find_contour_x_at_y(body_mask, shoulder_y, side='right')

        if left_x is not None and right_x is not None:
            true_left_pt  = (left_x,  shoulder_y)
            true_right_pt = (right_x, shoulder_y)

            # Красные точки — реальный край плеча по контуру
            cv2.circle(frame, true_left_pt,  10, (0, 0, 255), -1)
            cv2.circle(frame, true_right_pt, 10, (0, 0, 255), -1)

            cv2.putText(frame, "TRUE_L", (left_x + 8, shoulder_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "TRUE_R", (right_x - 80, shoulder_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Линия между реальными точками плеч — белая
            cv2.line(frame, true_left_pt, true_right_pt, (255, 255, 255), 2)

            # Ширина в пикселях
            true_shoulder_width_px = right_x - left_x
            mp_shoulder_width_px   = abs(rs_px[0] - ls_px[0])

            # Вывод на экран
            cv2.putText(frame,
                        f"TRUE width: {true_shoulder_width_px}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame,
                        f"MP   width: {mp_shoulder_width_px}px",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Разница в процентах
            if mp_shoulder_width_px > 0:
                diff_pct = (true_shoulder_width_px - mp_shoulder_width_px) / mp_shoulder_width_px * 100
                cv2.putText(frame,
                            f"Diff: +{diff_pct:.1f}%",
                            (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Рисуем истинные ширины плеч для справки
            cv2.putText(frame, f"Shoulder px: {true_shoulder_width_px}", (10, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ─────────────────────────────────
        # Бёдра — X из контура (внутри границ плеч), Y из MediaPipe
        # ─────────────────────────────────

        # Y бедер от MediaPipe (в пикселях) — используем среднее двух бедер
        # чтобы измерять ширину на одном горизонтальном уровне
        hip_y = int((lh.y + rh.y) / 2 * h)

        # УМНЫЙ МЕТОД: ищем края бедер только ВНУТРИ диапазона плеч
        # Туловище не расширяется очень сильно от плеч к бедрам
        # Это избегает путаницы с руками
        if left_x is not None and right_x is not None:
            left_hip_x  = find_contour_x_in_range(body_mask, hip_y, left_x, right_x, side='left')
            right_hip_x = find_contour_x_in_range(body_mask, hip_y, left_x, right_x, side='right')
        else:
            left_hip_x = None
            right_hip_x = None

        if left_hip_x is not None and right_hip_x is not None:
            true_left_hip_pt  = (left_hip_x,  hip_y)
            true_right_hip_pt = (right_hip_x, hip_y)

            # Красные точки — реальный край бедра по контуру
            cv2.circle(frame, true_left_hip_pt,  10, (0, 0, 255), -1)
            cv2.circle(frame, true_right_hip_pt, 10, (0, 0, 255), -1)

            cv2.putText(frame, "TRUE_L_HIP", (left_hip_x + 8, hip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "TRUE_R_HIP", (right_hip_x - 80, hip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Линия между реальными точками бедер — белая
            cv2.line(frame, true_left_hip_pt, true_right_hip_pt, (255, 255, 255), 2)

            # Ширина в пикселях
            true_hip_width_px = right_hip_x - left_hip_x
            mp_hip_width_px   = abs(rh_px[0] - lh_px[0])

            # Вывод на экран
            cv2.putText(frame,
                        f"TRUE hip width: {true_hip_width_px}px",
                        (10, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame,
                        f"MP   hip width: {mp_hip_width_px}px",
                        (10, 142),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Разница в процентах
            if mp_hip_width_px > 0:
                diff_pct = (true_hip_width_px - mp_hip_width_px) / mp_hip_width_px * 100
                cv2.putText(frame,
                            f"Hip diff: +{diff_pct:.1f}%",
                            (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ─────────────────────────────────
        # ----- Framemom Meter (на основе ratio) -----
        # Вычисляем ratio shoulder/hip ТОЛЬКО если оба значения есть
        # ─────────────────────────────────

        if left_x is not None and right_x is not None and left_hip_x is not None and right_hip_x is not None:
            true_shoulder_width_px = right_x - left_x
            true_hip_width_px = right_hip_x - left_hip_x

            if true_hip_width_px > 0:
                ratio = true_shoulder_width_px / true_hip_width_px

                # Выбираем состояние и цвет по ratio
                if ratio >= 1.1:
                    meter_state = 'david'
                    meter_label = 'David Laid'
                    meter_img = img_david
                    header_color = (0, 255, 0)      # зелёный для >= 1.1
                elif ratio >= 1.0:
                    meter_state = 'klav'
                    meter_label = 'KlaviKular (luxmaxer)'
                    meter_img = img_klav
                    header_color = (0, 215, 255)    # жёлто-оранжевый для 1.0-1.1
                else:
                    meter_state = 'chud'
                    meter_label = 'Chud'
                    meter_img = img_chud
                    header_color = (0, 0, 255)      # красный для < 1.0

                # Рисуем Framemog Meter
                header = "Framemog Meter"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(header, font, scale, thickness)
                header_x = max(10, (w - text_w) // 2)
                header_y = 30
                cv2.putText(frame, header, (header_x, header_y), font, scale, header_color, thickness)

                # Рисуем саму метку (label)
                cv2.putText(frame, meter_label, (10, 200), font, 0.8, (255, 255, 255), 2)

                # Рисуем ratio для справки
                cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 224), font, 0.5, (255, 255, 255), 1)

                # Наложим картинку в правый верхний угол
                if meter_img is not None:
                    frame = overlay_image(frame, meter_img, frame.shape[1] - 130, 10, scale=120 / max(1, meter_img.shape[1]))

    # ─────────────────────────────────────
    # Отображение
    # ─────────────────────────────────────

    cv2.imshow("FrameMog", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("Готово.")
