import cv2
import numpy as np
import pyautogui
import time
import win32gui, win32ui, win32con
import os
import keyboard
import sys
import math
from PIL import Image
from imagehash import phash, dhash
from typing import List, Tuple, Dict, Any, Optional, Set
import concurrent.futures

# --- 配置參數 ---
TARGET_WINDOW_TITLE = "BrownDust II" # 你的遊戲視窗標題

# 遊戲棋盤在遊戲視窗中的相對位置和大小
GAME_BOARD_OFFSET_X = 445 # 使用你最新測量的左上角 X 座標
GAME_BOARD_OFFSET_Y = 259 # 使用你最新測量的左上角 Y 座標

# 方塊的近似大小
TILE_WIDTH = 64
TILE_HEIGHT = 78

# 方塊之間的間隔 (從你的截圖看，可能為0)
TILE_GAP_X = 0
TILE_GAP_Y = 0

# 棋盤的行列數 (你已確認為 7 行 16 列)
BOARD_ROWS = 7
BOARD_COLS = 16

# 計算棋盤的總寬度和總高度
GAME_BOARD_WIDTH = BOARD_COLS * (TILE_WIDTH + TILE_GAP_X)
GAME_BOARD_HEIGHT = BOARD_ROWS * (TILE_HEIGHT + TILE_GAP_Y)


# 實體方塊的匹配閾值
MATCH_THRESHOLD_SOLID = 0.85
# 空白方塊的匹配閾值 (目前主要使用純色判斷，但仍保留此值用於哈希回退)
MATCH_THRESHOLD_EMPTY = 0.70

# 點擊之間的延遲（秒）
CLICK_DELAY = 0.1

# 掃描遊戲棋盤的間隔時間（秒）
SCAN_INTERVAL = 0.2

# --- 繪圖參數調整 ---
RECTANGLE_THICKNESS = 2 # 方塊邊框的粗細
LINE_THICKNESS = 4 # 消除連線的粗細
LINE_COLOR = (0, 255, 0) # 消除連線的顏色 (B, G, R) - 綠色
FONT_SCALE = 0.7 # 標籤文字的字體大小
TEXT_THICKNESS = 2 # 標籤文字的粗細

# --- 圖片處理與識別相關常量 ---
# 深藍綠色背景的BGR範圍 (用於純色判斷空方塊)
DEEP_BLUE_GREEN_MIN = np.array([30, 30, 0], dtype=np.uint8)   # 藍, 綠, 紅 (更低的紅色，更寬的藍綠)
DEEP_BLUE_GREEN_MAX = np.array([180, 160, 80], dtype=np.uint8) # 藍, 綠, 紅 (更高的藍綠上限)

# 紅色背景的BGR範圍 (新增：用於處理消除後的紅色底色)
RED_BACKGROUND_MIN = np.array([0, 0, 80], dtype=np.uint8)     # 低藍, 低綠, 高紅
RED_BACKGROUND_MAX = np.array([100, 100, 255], dtype=np.uint8) # 中藍, 中綠, 極高紅

# --- 除錯圖片保存設定 ---
SAVE_DEBUG_IMAGES = False # <--- 這裡控制是否保存除錯圖片 (True/False) - 默認修改為 False
DEBUG_IMAGES_FOLDER = "debug_board_scans" # <--- 圖片保存的資料夾名稱


# --- 輔助函數：判斷顏色是否在漸變背景範圍內 (使用通道比例法) ---
def is_deep_blue_green(bgr_color: np.ndarray) -> bool:
    """
    判斷是否為深藍綠色背景：藍、綠通道明顯大於紅色通道。
    """
    b, g, r = bgr_color
    # 藍綠色背景通常 B 和 G 較高，且 R 較低
    return (b > 30 and g > 30) and (b > r * 1.2) and (g > r * 1.2)

def is_red_background(bgr_color: np.ndarray) -> bool:
    """
    判斷是否為紅色/棕色漸變背景：紅色通道明顯大於藍、綠通道。
    根據使用者提供的範例 (R:74~133, G:33~42, B:21~33)
    """
    b, g, r = bgr_color # 注意 OpenCV 是 BGR
    # 紅色通道 R 必須具備一定強度，且比例明顯高於 B 和 G
    return (r > 50) and (r > g * 1.5) and (r > b * 1.5)

def is_any_background(bgr_color: np.ndarray) -> bool:
    """
    綜合判斷背景。
    """
    return is_deep_blue_green(bgr_color) or is_red_background(bgr_color)

# --- 輔助函數：判斷圖片是否接近純色 (使用 9 宮格判斷) ---
def is_nearly_solid_color(cv2_img: np.ndarray, tolerance: int) -> Tuple[bool, Optional[np.ndarray], List[np.ndarray]]:
    """
    判斷一個圖像區域是否接近純色，通過將其分為 9 個子區域並比較其平均顏色。
    返回 (是否為純色, 整體平均顏色, 9個子區域平均顏色列表)。
    """
    if cv2_img.ndim == 2:
        img_check = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
    elif cv2_img.shape[2] == 4:
        img_check = cv2.cvtColor(cv2_img, cv2.COLOR_RGBA2BGR)
    else:
        img_check = cv2_img

    h, w, _ = img_check.shape

    sub_region_avg_colors_list = []

    # 處理極小圖像的情況，避免除以零或無效裁剪
    if h < 3 or w < 3: 
        if h * w > 0:
            std_dev_color = np.std(img_check.astype(float), axis=(0, 1))
            max_val = np.max(img_check, axis=(0, 1))
            min_val = np.min(img_check, axis=(0, 1))
            overall_mean = np.mean(img_check, axis=(0, 1))
            sub_region_avg_colors_list.append(overall_mean)
            if np.all(std_dev_color < tolerance) and np.all((max_val - min_val) < tolerance * 2):
                return True, overall_mean, sub_region_avg_colors_list
        return False, None, sub_region_avg_colors_list

    sub_h = h // 3
    sub_w = w // 3

    for r_idx in range(3):
        for c_idx in range(3):
            y_start = r_idx * sub_h
            y_end = y_start + sub_h
            x_start = c_idx * sub_w
            x_end = c_idx + sub_w

            # 確保覆蓋邊緣像素
            if r_idx == 2: y_end = h
            if c_idx == 2: x_end = w

            sub_img = img_check[y_start:y_end, x_start:x_end]
            
            if sub_img.size == 0:
                return False, None, sub_region_avg_colors_list 

            avg_color = np.mean(sub_img, axis=(0, 1))
            sub_region_avg_colors_list.append(avg_color)

    if not sub_region_avg_colors_list or len(sub_region_avg_colors_list) != 9:
        return False, None, sub_region_avg_colors_list

    avg_colors_np = np.array(sub_region_avg_colors_list)
    max_avg_color = np.max(avg_colors_np, axis=0)
    min_avg_color = np.min(avg_colors_np, axis=0)

    if np.all((max_avg_color - min_avg_color) < tolerance):
        return True, np.mean(img_check, axis=(0, 1)), sub_region_avg_colors_list 
    
    return False, np.mean(img_check, axis=(0, 1)), sub_region_avg_colors_list 

class LianLianKanBot:
    def __init__(self, template_folder="accumulated_templates_library"): 
        self.template_folder = template_folder
        self.BOARD_ROWS = BOARD_ROWS
        self.BOARD_COLS = BOARD_COLS
        self.BOARD_TILE_WIDTH = TILE_WIDTH
        self.BOARD_TILE_HEIGHT = TILE_HEIGHT
        self.BOARD_TILE_GAP_X = TILE_GAP_X
        self.BOARD_TILE_GAP_Y = TILE_GAP_Y


        # 遊戲棋盤的偏移量 (從全螢幕左上角開始計算)
        self.GAME_BOARD_OFFSET_X = GAME_BOARD_OFFSET_X
        self.GAME_BOARD_OFFSET_Y = GAME_BOARD_OFFSET_Y
        self.GAME_BOARD_WIDTH = GAME_BOARD_WIDTH
        self.GAME_BOARD_HEIGHT = GAME_BOARD_HEIGHT

        # 裁剪的內部區域參數 (需要與模板生成器中的設置一致)
        self.INNER_TILE_PADDING = 8 
        self.PRE_PURE_COLOR_CHECK_PADDING = 5 # 用於純色判斷的區域從 tile_img_cropped 再次內縮的像素

        # 以下尺寸值應根據模板生成器的最終輸出進行設置
        self.NORMAL_TEMPLATE_WIDTH = 48 
        self.NORMAL_TEMPLATE_HEIGHT = 62 
        self.PURE_COLOR_CHECK_WIDTH = self.NORMAL_TEMPLATE_WIDTH - (2 * self.PRE_PURE_COLOR_CHECK_PADDING) 
        self.PURE_COLOR_CHECK_HEIGHT = self.NORMAL_TEMPLATE_HEIGHT - (2 * self.PRE_PURE_COLOR_CHECK_PADDING) 
        
        # 哈希閾值和純色判斷容忍度
        self.HASH_THRESHOLD = 20 # 提高閾值以增加辨識容忍度
        self.EMPTY_TILE_HASH_THRESHOLD = 22 
        self.EMPTY_TILE_COLOR_TOLERANCE = 25 

        # Load templates
        self.templates: Dict[str, Tuple[Image.Image, Any, np.ndarray]] = self._load_templates(template_folder)
        print(f"Loaded {len(self.templates)} general templates from {template_folder}")

        # 確保空模板和鎖定模板的哈希值被單獨儲存
        self.empty_blue_green_template_hash: Optional[Any] = None # 深藍綠色空模板哈希
        self.empty_red_striped_template_hash: Optional[Any] = None # 紅色條紋空模板哈希
        self.locked_template_hashes: List[Any] = []

        # 加載深藍綠色空模板
        blue_green_empty_path = os.path.join(template_folder, "template_empty_blue_green.png")
        if os.path.exists(blue_green_empty_path):
            try:
                self.empty_blue_green_template_hash = phash(Image.open(blue_green_empty_path))
                print("Loaded 'template_empty_blue_green.png'.")
            except Exception as e:
                print(f"WARNING: Could not load blue-green empty template hash: {e}")

        # 加載紅色條紋空模板
        red_striped_empty_path = os.path.join(template_folder, "template_empty_red_striped.png")
        if os.path.exists(red_striped_empty_path):
            try:
                self.empty_red_striped_template_hash = phash(Image.open(red_striped_empty_path))
                print("Loaded 'template_empty_red_striped.png'.")
            except Exception as e:
                print(f"WARNING: Could not load red-striped empty template hash: {e}")

        # 加載所有鎖定模板
        for i in range(1, 5): 
            locked_path = os.path.join(template_folder, f"template_locked_{i}.png")
            if os.path.exists(locked_path):
                try:
                    self.locked_template_hashes.append(phash(Image.open(locked_path)))
                except Exception as e:
                    print(f"WARNING: Could not load locked template hash {i}: {e}")
        print(f"Loaded {len(self.locked_template_hashes)} locked templates and {('2' if self.empty_blue_green_template_hash and self.empty_red_striped_template_hash else '1' if self.empty_blue_green_template_hash or self.empty_red_striped_template_hash else '0')} empty templates.")


    def _load_templates(self, template_folder: str) -> Dict[str, Tuple[Image.Image, Any, np.ndarray]]:
        """加載模板圖片並計算其感知哈希值。"""
        templates = {}
        for filename in os.listdir(template_folder):
            # 篩選掉特殊模板，只加載普通圖標模板
            if filename.endswith(".png") and not (filename.startswith("template_empty") or filename.startswith("template_locked_")):
                template_path = os.path.join(template_folder, filename)
                try:
                    pil_img = Image.open(template_path)
                    p_hash = phash(pil_img)
                    cv2_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                    
                    # 從文件名中提取 ID (例如 "template_001.png" -> "001")
                    template_id = filename.replace("template_", "").replace(".png", "")
                    templates[template_id] = (pil_img, p_hash, cv2_img)
                except Exception as e:
                    print(f"❌ 警告: 無法加載模板圖片或計算哈希: {template_path} - {e}")
        return templates

    def _get_game_board_tiles(self, game_window_img: np.ndarray) -> Tuple[List[List[str]], Dict[Tuple[int, int], Tuple[int, int]], np.ndarray]:
        """
        從遊戲視窗截圖中識別棋盤上的方塊。
        返回 (board_matrix, tile_centers, debug_img_base)。
        debug_img_base 是基於原始識別結果繪製的圖像。
        """
        board_matrix = [["" for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
        tile_centers = {} # 儲存每個可點擊方塊的中心絕對座標 (相對於全螢幕)
        
        # 複製一份用於繪製原始識別結果的圖像
        debug_img_base = game_window_img.copy()

        board_img = game_window_img[
            self.GAME_BOARD_OFFSET_Y : self.GAME_BOARD_OFFSET_Y + self.GAME_BOARD_HEIGHT,
            self.GAME_BOARD_OFFSET_X : self.GAME_BOARD_OFFSET_X + self.GAME_BOARD_WIDTH
        ]

        if board_img.size == 0:
            print("❌ 截取的棋盤區域為空，請檢查棋盤座標設定。")
            return board_matrix, tile_centers, debug_img_base

        # 這裡增加一個內部計數器，用於每個方塊的命名
        tile_scan_count = 0 

        # 確保未知方塊的保存資料夾存在
        unknown_tiles_folder = os.path.join(globals()['DEBUG_IMAGES_FOLDER'], "unknown_tiles")
        if not os.path.exists(unknown_tiles_folder):
            os.makedirs(unknown_tiles_folder)


        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                x = c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X)
                y = r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y)

                tile_img = board_img[y : y + self.BOARD_TILE_HEIGHT, x : x + self.BOARD_TILE_WIDTH]

                if tile_img.size == 0:
                    board_matrix[r][c] = "空ROI"
                    # 在 debug_img_base 上繪製
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (100, 100, 100), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img_base, "E_ROI", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (100, 100, 100), TEXT_THICKNESS)
                    continue
                
                cropped_y_start = self.INNER_TILE_PADDING
                cropped_y_end = self.INNER_TILE_PADDING + self.NORMAL_TEMPLATE_HEIGHT
                cropped_x_start = self.INNER_TILE_PADDING
                cropped_x_end = self.INNER_TILE_PADDING + self.NORMAL_TEMPLATE_WIDTH

                if not (0 <= cropped_y_start < cropped_y_end <= tile_img.shape[0] and
                        0 <= cropped_x_start < cropped_x_end <= tile_img.shape[1]):
                    board_matrix[r][c] = "裁剪錯誤"
                    # 在 debug_img_base 上繪製
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 0, 255), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img_base, "CROP_ERR", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (0, 0, 255), TEXT_THICKNESS)
                    continue

                tile_img_cropped = tile_img[
                    cropped_y_start : cropped_y_end,
                    cropped_x_start : cropped_x_end
                ]

                # --- 保存每個裁剪後的方塊圖片 (用於除錯，無論識別結果如何) ---
                if globals()['SAVE_DEBUG_IMAGES']: 
                    tile_scan_count += 1
                    batch_id = globals()['scan_count']
                    tile_save_name = f"scan_{batch_id:05d}_r{r:02d}_c{c:02d}.png"
                    tile_save_path = os.path.join(globals()['DEBUG_IMAGES_FOLDER'], "tiles", tile_save_name)
                    
                    # 確保 "tiles" 子資料夾存在
                    tile_folder = os.path.join(globals()['DEBUG_IMAGES_FOLDER'], "tiles")
                    if not os.path.exists(tile_folder):
                        os.makedirs(tile_folder)

                    try:
                        cv2.imwrite(tile_save_path, tile_img_cropped)
                    except Exception as e:
                        print(f"❌ 保存單個方塊圖片失敗: {tile_save_path} - {e}")

    def _process_single_tile(self, r, c, tile_img_cropped, scan_count):
        """處理單個方塊的識別邏輯，適合平行化調用。"""
        if tile_img_cropped.size == 0:
            return r, c, "裁剪錯誤", None

        # 轉換為 PIL 圖片計算雜湊
        pil_tile_img = Image.fromarray(cv2.cvtColor(tile_img_cropped, cv2.COLOR_BGR2RGB))
        current_tile_hash = phash(pil_tile_img)

        matched_id = "未知"
        
        # 1. 鎖定模板匹配
        for i, locked_hash in enumerate(self.locked_template_hashes):
            if (current_tile_hash - locked_hash) <= self.HASH_THRESHOLD:
                return r, c, f"lock{i+1}", (self.GAME_BOARD_OFFSET_X + c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X) + self.BOARD_TILE_WIDTH // 2,
                                          self.GAME_BOARD_OFFSET_Y + r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y) + self.BOARD_TILE_HEIGHT // 2)

        # 2. 普通模板匹配
        best_hash_distance = float('inf')
        for template_id_str, (_, template_hash_obj, _) in self.templates.items():
            distance = current_tile_hash - template_hash_obj
            if distance <= self.HASH_THRESHOLD and distance < best_hash_distance:
                best_hash_distance = distance
                matched_id = template_id_str
        
        if matched_id != "未知":
            return r, c, matched_id, (self.GAME_BOARD_OFFSET_X + c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X) + self.BOARD_TILE_WIDTH // 2,
                                     self.GAME_BOARD_OFFSET_Y + r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y) + self.BOARD_TILE_HEIGHT // 2)

        # 3. 空方塊判定
        is_empty_tile = False
        pure_check_y_start = self.PRE_PURE_COLOR_CHECK_PADDING
        pure_check_y_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_HEIGHT
        pure_check_x_start = self.PRE_PURE_COLOR_CHECK_PADDING
        pure_check_x_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_WIDTH

        pure_color_check_area = None
        if (0 <= pure_check_y_start < pure_check_y_end <= tile_img_cropped.shape[0] and
            0 <= pure_check_x_start < pure_check_x_end <= tile_img_cropped.shape[1]):
            pure_color_check_area = tile_img_cropped[pure_check_y_start:pure_check_y_end, pure_check_x_start:pure_check_x_end]
        else:
            pure_color_check_area = tile_img_cropped

        if pure_color_check_area.size > 0:
            is_nearly_solid, overall_mean_color, _ = is_nearly_solid_color(pure_color_check_area, self.EMPTY_TILE_COLOR_TOLERANCE)
            if is_nearly_solid and overall_mean_color is not None and is_any_background(overall_mean_color):
                is_empty_tile = True

        if not is_empty_tile and self.empty_red_striped_template_hash and \
           (current_tile_hash - self.empty_red_striped_template_hash) <= self.EMPTY_TILE_HASH_THRESHOLD:
            is_empty_tile = True
        
        if not is_empty_tile and self.empty_blue_green_template_hash and \
           (current_tile_hash - self.empty_blue_green_template_hash) <= self.EMPTY_TILE_HASH_THRESHOLD:
            is_empty_tile = True

        if is_empty_tile:
            return r, c, "empty", None
        
        # 4. 未知方塊保存 (調試用)
        if globals()['SAVE_DEBUG_IMAGES']:
            # 這裡儘量減少 I/O 阻塞，或者考慮異步寫入
            pass

        return r, c, "未知", None

    def _get_game_board_tiles(self, full_screenshot_cv2):
        """優化後的棋盤掃描，使用多執行緒平行處理方塊。"""
        board_matrix = [["" for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
        tile_centers = {}
        # 始終產生 debug_img_base，因為 main 迴圈需要它來顯示視窗
        debug_img_base = full_screenshot_cv2.copy()

        tile_tasks = []
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                x = c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X)
                y = r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y)
                tile_img_cropped = full_screenshot_cv2[
                    self.GAME_BOARD_OFFSET_Y + y : self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT,
                    self.GAME_BOARD_OFFSET_X + x : self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH
                ]
                tile_tasks.append((r, c, tile_img_cropped))

        # 使用 ThreadPoolExecutor 平行處理識別邏輯
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_tile = {executor.submit(self._process_single_tile, r, c, img, globals()['scan_count']): (r, c) for r, c, img in tile_tasks}
            for future in concurrent.futures.as_completed(future_to_tile):
                r, c, res_id, center = future.result()
                board_matrix[r][c] = res_id
                if center:
                    tile_centers[(r, c)] = center

        # 繪製識別標籤
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                x = self.GAME_BOARD_OFFSET_X + c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X)
                y = self.GAME_BOARD_OFFSET_Y + r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y)
                label = board_matrix[r][c]
                color = (0, 255, 0) if label == "empty" else (255, 0, 0) if "lock" in label else (0, 255, 255)
                cv2.rectangle(debug_img_base, (x, y), (x + self.BOARD_TILE_WIDTH, y + self.BOARD_TILE_HEIGHT), color, 1)

        return board_matrix, tile_centers, debug_img_base


    def dynamic_classify_scan(self, full_screenshot_cv2):
        """優化後的即時分類模式：結合感知雜湊與顏色特徵進行精準分類。"""
        # 初始嘗試參數
        current_threshold = 14 
        COLOR_DISTANCE_THRESHOLD = 35 # BGR 顏色向量距離閾值 (可調整)
        max_attempts = 3
        
        for attempt in range(max_attempts):
            board_matrix = [["empty" for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
            tile_centers = {}
            seen_tile_features = [] # [(hash, avg_bgr, temp_id), ...]
            next_temp_id = 1
            
            # 1. 準備所有方塊
            for r in range(self.BOARD_ROWS):
                for c in range(self.BOARD_COLS):
                    x = c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X)
                    y = r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y)
                    tile_raw = full_screenshot_cv2[
                        self.GAME_BOARD_OFFSET_Y + y : self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT,
                        self.GAME_BOARD_OFFSET_X + x : self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH
                    ]
                    
                    if tile_raw.size == 0: continue

                    # --- 強化的空位判定 ---
                    padding_inner = 15
                    tile_inner = tile_raw[padding_inner:-padding_inner, padding_inner:-padding_inner]
                    
                    is_empty = False
                    avg_bgr = None
                    if tile_inner.size > 0:
                        is_solid, avg_bgr, _ = is_nearly_solid_color(tile_inner, self.EMPTY_TILE_COLOR_TOLERANCE)
                        if is_solid and is_any_background(avg_bgr):
                            is_empty = True
                    
                    if not is_empty:
                        pil_raw = Image.fromarray(cv2.cvtColor(tile_raw, cv2.COLOR_BGR2RGB))
                        h_raw = phash(pil_raw)
                        if self.empty_red_striped_template_hash and (h_raw - self.empty_red_striped_template_hash) <= 22:
                            is_empty = True
                        elif self.empty_blue_green_template_hash and (h_raw - self.empty_blue_green_template_hash) <= 22:
                            is_empty = True

                    if is_empty:
                        board_matrix[r][c] = "empty"
                        continue

                    # --- 顏色特徵提取 (BGR 平均值) ---
                    # 如果前面 is_nearly_solid_color 沒算過 avg_bgr，這裡補算
                    if avg_bgr is None:
                        avg_bgr = np.mean(tile_inner if tile_inner.size > 0 else tile_raw, axis=(0, 1))

                    # --- 影像預處理 (用於雜湊) ---
                    gray = cv2.cvtColor(tile_raw, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    pil_prep = Image.fromarray(blurred)
                    h = phash(pil_prep)
                    
                    # --- 雙重比對邏輯：雜湊 + 顏色 ---
                    found_match = False
                    for seen_h, seen_bgr, tid in seen_tile_features:
                        h_dist = h - seen_h
                        # 計算 BGR 空間的歐幾里得距離
                        c_dist = np.linalg.norm(avg_bgr - seen_bgr)
                        
                        if h_dist <= current_threshold and c_dist <= COLOR_DISTANCE_THRESHOLD:
                            board_matrix[r][c] = tid
                            found_match = True
                            break
                    
                    if not found_match:
                        new_id = f"T{next_temp_id:02d}"
                        seen_tile_features.append((h, avg_bgr, new_id))
                        board_matrix[r][c] = new_id
                        next_temp_id += 1
                    
                    abs_x = self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH // 2
                    abs_y = self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2
                    tile_centers[(r, c)] = (abs_x, abs_y)

            # 檢查分類結果是否合理
            if len(seen_tile_features) > 40 and attempt < max_attempts - 1:
                current_threshold += 4 
                COLOR_DISTANCE_THRESHOLD += 10
                continue
            else:
                break 

        # 3. 繪製 Debug 資訊
        debug_img = full_screenshot_cv2.copy()
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                x = self.GAME_BOARD_OFFSET_X + c * (self.BOARD_TILE_WIDTH + self.BOARD_TILE_GAP_X)
                y = self.GAME_BOARD_OFFSET_Y + r * (self.BOARD_TILE_HEIGHT + self.BOARD_TILE_GAP_Y)
                label = board_matrix[r][c]
                if label != "empty":
                    cv2.rectangle(debug_img, (x, y), (x + self.BOARD_TILE_WIDTH, y + self.BOARD_TILE_HEIGHT), (255, 0, 255), 1)
                    cv2.putText(debug_img, label, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return board_matrix, tile_centers, debug_img

    def find_min_bends(self, board, p_start, p_end):
        """使用 BFS 尋找從 p_start 到 p_end 的最小轉彎次數。"""
        from collections import deque
        r_start, c_start = p_start
        r_end, c_end = p_end
        
        # queue 存儲: (r, c, current_bends, direction)
        # direction: 0:None, 1:Up, 2:Down, 3:Left, 4:Right
        queue = deque([(r_start, c_start, 0, 0)])
        visited = {} # (r, c, dir) -> min_bends
        
        directions = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]
        
        while queue:
            r, c, bends, last_dir = queue.popleft()
            if (r, c) == (r_end, c_end): return bends
            if bends > 2: continue
                
            for dr, dc, d_code in directions:
                nr, nc = r + dr, c + dc
                new_bends = bends + (1 if last_dir != 0 and d_code != last_dir else 0)
                if new_bends > 2: continue
                
                # 邊界檢查：允許棋盤外圍一圈行走 (-1 到 ROWS/COLS)
                if not (-1 <= nr <= self.BOARD_ROWS and -1 <= nc <= self.BOARD_COLS): continue
                
                # 通行檢查：如果是棋盤內且不是目標點，必須是空的
                if 0 <= nr < self.BOARD_ROWS and 0 <= nc < self.BOARD_COLS:
                    if (nr, nc) != (r_end, c_end) and board[nr][nc] != "empty": continue
                
                state = (nr, nc, d_code)
                if state not in visited or visited[state] > new_bends:
                    visited[state] = new_bends
                    queue.append((nr, nc, new_bends, d_code))
        return 99

    def get_all_eliminable_pairs(self, board):
        """一次性找出棋盤上所有可消除的對子。"""
        pairs_by_id = {}
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                tid = board[r][c]
                if tid not in ["empty", "lock1", "lock2", "lock3", "lock4", "未知", "裁剪錯誤", "空ROI"]:
                    if tid not in pairs_by_id: pairs_by_id[tid] = []
                    pairs_by_id[tid].append((r, c))

        all_pairs = []
        for tid, coords in pairs_by_id.items():
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    p1, p2 = coords[i], coords[j]
                    bends = self.find_min_bends(board, p1, p2)
                    if bends <= 2: all_pairs.append((bends, p1, p2))
        
        all_pairs.sort(key=lambda x: x[0]) # 直線優先
        return [(p1, p2) for b, p1, p2 in all_pairs]


# --- 全域變數 for 視窗資訊和腳本狀態 ---
game_window_x, game_window_y = 0, 0
running = False
display_window_name = "Lianliankan Bot Debug"
last_display_img = None
bot_instance: Optional[LianLianKanBot] = None
scan_count = 0 # 記錄掃描次數，用於檔案命名

# 棋盤狀態變數
current_raw_board_matrix: List[List[str]] = [] # 這一輪 _get_game_board_tiles 的原始掃描結果
last_raw_board_matrix: List[List[str]] = []    # 上一輪 _get_game_board_tiles 的原始掃描結果
persisted_e_board: List[List[str]] = []       # 持久化的 E 棋盤，只記錄 "E" 或 ""

# 新增：重複偵測與模式控制變數
last_clicked_pairs_coords = set() # 記錄上一輪點擊過的座標對
current_scan_mode = 'dynamic'     # 預設模式：'dynamic' (動態分類) 或 'template' (模板匹配)

# --- 熱鍵回調函數 ---
def start_bot():
    global running, current_scan_mode
    running = True
    current_scan_mode = 'dynamic' # 啟動時重置為預設的動態模式
    print("\n--- 外掛已啟動 (按 F4 暫停, Q 鍵退出) ---")

def stop_bot():
    global running
    running = False
    print("\n--- 外掛已暫停 (按 F3 啟動, Q 鍵退出) ---")

def exit_program():
    global running
    running = False
    print("\n--- 程式已由使用者退出 ---")
    os._exit(0)

def toggle_save_debug_images():
    global SAVE_DEBUG_IMAGES, DEBUG_IMAGES_FOLDER
    SAVE_DEBUG_IMAGES = not SAVE_DEBUG_IMAGES
    print(f"\n--- 除錯圖片保存已 {'開啟' if SAVE_DEBUG_IMAGES else '關閉'} ---")
    if SAVE_DEBUG_IMAGES:
        # 確保資料夾存在，如果開啟了保存功能
        if not os.path.exists(DEBUG_IMAGES_FOLDER):
            os.makedirs(DEBUG_IMAGES_FOLDER)
            print(f"已創建資料夾: {DEBUG_IMAGES_FOLDER}")
        
        # 清空舊的除錯圖片，避免累積過多
        print(f"清空舊的除錯圖片在 {DEBUG_IMAGES_FOLDER}...")
        # 同時清空 tiles 和 unknown_tiles 子資料夾
        for sub_folder in ["tiles", "unknown_tiles"]:
            folder_path = os.path.join(DEBUG_IMAGES_FOLDER, sub_folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    try:
                        os.remove(os.path.join(folder_path, f))
                    except OSError as e:
                        print(f"錯誤: 無法刪除檔案 {f} - {e}")
                try:
                    os.rmdir(folder_path) # 移除資料夾本身
                except OSError as e:
                    print(f"錯誤: 無法刪除資料夾 {folder_path} - {e}")
            # 重新創建資料夾 (確保存在)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # 清空主資料夾裡的 .png 圖片
        for f in os.listdir(DEBUG_IMAGES_FOLDER):
            if f.endswith(".png"): # 只刪除 .png 文件
                try:
                    os.remove(os.path.join(DEBUG_IMAGES_FOLDER, f))
                except OSError as e:
                    print(f"錯誤: 無法刪除檔案 {f} - {e}")
        print("舊圖片已清空。")
    
# --- 主程式邏輯 ---
def main():
    global game_window_x, game_window_y, running, last_display_img, bot_instance, scan_count
    global SAVE_DEBUG_IMAGES, DEBUG_IMAGES_FOLDER 
    # 新增三個棋盤和集合的全域變數的引用
    global current_raw_board_matrix, last_raw_board_matrix, persisted_e_board
    global last_clicked_pairs_coords, current_scan_mode

    print("連連看點擊外掛準備中...")
    print("按 F3 啟動外掛，按 F4 暫停外掛，按 F5 開關除錯圖片保存，按 Q 鍵退出程式。")
    print("程式運行時，會彈出一個顯示識別結果的視窗，它會覆蓋在遊戲上方。")

    # 註冊熱鍵
    keyboard.add_hotkey('f3', start_bot)
    keyboard.add_hotkey('f4', stop_bot)
    keyboard.add_hotkey('f5', toggle_save_debug_images)
    keyboard.add_hotkey('q', exit_program)

    # 尋找遊戲視窗
    hwnd = win32gui.FindWindow(None, TARGET_WINDOW_TITLE)
    while not hwnd:
        print(f"❌ 無法找到遊戲視窗: '{TARGET_WINDOW_TITLE}'，請確認遊戲是否已開啟且標題正確！")
        time.sleep(1)
        hwnd = win32gui.FindWindow(None, TARGET_WINDOW_TITLE)
    print(f"✅ 找到遊戲視窗: '{TARGET_WINDOW_TITLE}'")

    # 初始化 Bot 實例
    bot_instance = LianLianKanBot(template_folder="accumulated_templates_library") 

    # Bot 啟動時檢查 SAVE_DEBUG_IMAGES 的狀態
    if SAVE_DEBUG_IMAGES: 
        if not os.path.exists(DEBUG_IMAGES_FOLDER):
            os.makedirs(DEBUG_IMAGES_FOLDER)
        print(f"--- 除錯圖片保存已 開啟 ---")
        print(f"清空舊的除錯圖片在 {DEBUG_IMAGES_FOLDER}...")
        
        for sub_folder in ["tiles", "unknown_tiles"]:
            folder_path = os.path.join(DEBUG_IMAGES_FOLDER, sub_folder)
            if os.path.exists(folder_path):
                for f in os.listdir(folder_path):
                    try:
                        os.remove(os.path.join(folder_path, f))
                    except OSError as e:
                        print(f"錯誤: 無法刪除檔案 {f} - {e}")
                try:
                    os.rmdir(folder_path)
                except OSError as e:
                    print(f"錯誤: 無法刪除資料夾 {folder_path} - {e}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        for f in os.listdir(DEBUG_IMAGES_FOLDER):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(DEBUG_IMAGES_FOLDER, f))
                except OSError as e:
                    print(f"錯誤: 無法刪除檔案 {f} - {e}")
        print("舊圖片已清空。")
    else:
        print(f"--- 除錯圖片保存已 關閉 (按 F5 開啟) ---")

    # 初始化棋盤狀態
    # 確保所有棋盤都被初始化為正確的尺寸和預設值
    current_raw_board_matrix = [["" for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)] 
    last_raw_board_matrix = [["" for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]   
    persisted_e_board = [["" for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]       

    # 創建用於顯示的 OpenCV 視窗
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(display_window_name, cv2.WND_PROP_TOPMOST, 1)

    # 調整 Debug 視窗大小和位置
    display_width = 1200
    display_height = 700
    display_pos_x = 1500
    display_pos_y = 0
    cv2.resizeWindow(display_window_name, display_width, display_height)
    cv2.moveWindow(display_window_name, display_pos_x, display_pos_y)

    while True:
        if running:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                game_window_x, game_window_y, x2, y2 = rect
                
                full_screenshot_pil = pyautogui.screenshot()
                img_np_rgb = np.array(full_screenshot_pil)
                img_full_screen_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

                img_game_window = img_full_screen_bgr[
                    game_window_y : y2,
                    game_window_x : x2
                ]

                if img_game_window.size == 0:
                    print("❌ 截取的遊戲視窗區域為空，請檢查 win32gui.GetWindowRect 返回的座標是否正確。")
                    print("這可能是因為遊戲視窗不在螢幕上，或者被最小化/隱藏了。")
                    time.sleep(SCAN_INTERVAL * 2)
                    continue
                
                scan_count += 1 # 每次掃描時增加計數器
                
                # --- START: 棋盤狀態管理邏輯 ---

                # 1. 將本輪的 current_raw_board_matrix 存為上一輪的 last_raw_board_matrix
                last_raw_board_matrix = [row[:] for row in current_raw_board_matrix] 

                # 2. 根據當前模式進行掃描
                if current_scan_mode == 'dynamic':
                    print(f"\n[{scan_count}] [動態分類模式] 掃描中...")
                    current_raw_board_matrix, tile_centers, display_img = bot_instance.dynamic_classify_scan(img_game_window)
                    solid_count = sum(1 for row in current_raw_board_matrix for val in row if "T" in val)
                    print(f"✨ 找到 {solid_count} 個方塊分類 (動態)。")
                else: # template mode
                    print(f"\n[{scan_count}] [模板匹配模式] 掃描中...")
                    current_raw_board_matrix, tile_centers, display_img = bot_instance._get_game_board_tiles(img_game_window)
                    solid_count = sum(1 for row in current_raw_board_matrix for val in row if val not in ["empty", "未知", "裁剪錯誤", "空ROI", ""])
                    print(f"✅ 找到 {solid_count} 個方塊 (模板)。")

                # 3. 根據 last_raw_board_matrix 和 current_raw_board_matrix 更新 persisted_e_board
                non_solid_states = ["", "empty", "未知", "裁剪錯誤", "空ROI"]

                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        last_state = last_raw_board_matrix[r][c]
                        current_state = current_raw_board_matrix[r][c]
                        is_last_solid = last_state not in non_solid_states
                        is_current_non_solid = current_state in non_solid_states

                        if is_last_solid and is_current_non_solid:
                            persisted_e_board[r][c] = "E"
                        elif current_state not in non_solid_states:
                            persisted_e_board[r][c] = ""
                        
                # 4. 創建 board_to_process
                board_to_process = [row[:] for row in current_raw_board_matrix] 

                # 5. 將 persisted_e_board 的 E 狀態應用
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        if persisted_e_board[r][c] == "E":
                            board_to_process[r][c] = "empty"
                            x_draw = c * (bot_instance.BOARD_TILE_WIDTH + bot_instance.BOARD_TILE_GAP_X)
                            y_draw = r * (bot_instance.BOARD_TILE_HEIGHT + bot_instance.BOARD_TILE_GAP_Y)
                            cv2.rectangle(display_img,
                                          (bot_instance.GAME_BOARD_OFFSET_X + x_draw, bot_instance.GAME_BOARD_OFFSET_Y + y_draw),
                                          (bot_instance.GAME_BOARD_OFFSET_X + x_draw + bot_instance.BOARD_TILE_WIDTH, bot_instance.GAME_BOARD_OFFSET_Y + y_draw + bot_instance.BOARD_TILE_HEIGHT),
                                          (0, 255, 0), RECTANGLE_THICKNESS)
                            cv2.putText(display_img, "E", (bot_instance.GAME_BOARD_OFFSET_X + x_draw + (bot_instance.BOARD_TILE_WIDTH - cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                            bot_instance.GAME_BOARD_OFFSET_Y + y_draw + (cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1] + 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
                                        
                # --- END: 棋盤狀態管理邏輯 ---

                # --- 尋找所有可消除的對子 ---
                eliminable_pairs = bot_instance.get_all_eliminable_pairs(board_to_process)

                # --- 重複偵測邏輯 ---
                current_pairs_coords = {frozenset([p1, p2]) for p1, p2 in eliminable_pairs}
                stuck_pairs = current_pairs_coords.intersection(last_clicked_pairs_coords)
                stuck_count = len(stuck_pairs)
                
                # 使用者邏輯：偵測到 > 5 對卡死 (即 >= 6 對)，才切換模式
                if stuck_count > 5:
                    print(f"⚠️ 偵測到 {stuck_count} 對重複出現且未被消除 (>5)，判定為誤判卡死。")
                    if current_scan_mode == 'dynamic':
                        print("🔄 正在從 [動態] 切換至 [模板] 模式以嘗試解卡...")
                        current_scan_mode = 'template'
                    else:
                        print("🔄 正在從 [模板] 切換回 [動態] 模式以嘗試解卡...")
                        current_scan_mode = 'dynamic'
                    
                    last_clicked_pairs_coords.clear() # 清空歷史，避免連續觸發
                    continue # 跳過本次點擊，直接進入下一輪使用新模式掃描
                
                # --- 執行點擊 ---
                if eliminable_pairs:
                    print(f"本輪找到 {len(eliminable_pairs)} 對可消除的方塊，準備點擊...")
                    clicked_this_round = set()
                    # 更新 last_clicked_pairs_coords，為下一輪檢測做準備
                    last_clicked_pairs_coords = current_pairs_coords

                    for p1, p2 in eliminable_pairs:
                        # 防止同一輪重複點擊已消除的方塊
                        if p1 in clicked_this_round or p2 in clicked_this_round:
                            continue

                        tile_name = board_to_process[p1[0]][p1[1]]
                        print(f"✅ 消除對子: {tile_name} 在 {p1} 和 {p2}")

                        if p1 in tile_centers and p2 in tile_centers:
                            draw_x1, draw_y1 = tile_centers[p1]
                            draw_x2, draw_y2 = tile_centers[p2]
                            
                            cv2.line(display_img, (draw_x1 - game_window_x, draw_y1 - game_window_y), (draw_x2 - game_window_x, draw_y2 - game_window_y), LINE_COLOR, LINE_THICKNESS)
                            
                            pyautogui.click(draw_x1, draw_y1)
                            time.sleep(CLICK_DELAY)
                            pyautogui.click(draw_x2, draw_y2)
                            time.sleep(CLICK_DELAY)
                            
                            clicked_this_round.add(p1)
                            clicked_this_round.add(p2)
                        else:
                            print(f"⚠️ 警告: 嘗試點擊的方塊 {p1} 或 {p2} 不在 tile_centers 中，跳過此對。")

                    time.sleep(0.5) 
                else:
                    print("本輪未找到可消除的對子。")
                    last_clicked_pairs_coords = set()
                    
                display_img_resized = cv2.resize(display_img, (display_width, display_height))
                cv2.imshow(display_window_name, display_img_resized)
                last_display_img = display_img.copy()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                time.sleep(SCAN_INTERVAL)

            except Exception as e:
                print(f"運行過程中發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        else:
            if last_display_img is not None:
                display_img_resized = cv2.resize(last_display_img, (display_width, display_height))
                cv2.imshow(display_window_name, display_img_resized)
            else:
                black_image = np.zeros((display_height, display_width, 3), np.uint8)
                cv2.putText(black_image, "Bot Paused (Press F3 to Start)", (50, display_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(display_window_name, black_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            time.sleep(0.5)

    cv2.destroyAllWindows()
    keyboard.unhook_all()

if __name__ == "__main__":
    main()