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
from typing import List, Tuple, Dict, Any, Optional

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
DEEP_BLUE_GREEN_MIN = np.array([65, 60, 15], dtype=np.uint8)   # 藍, 綠, 紅
DEEP_BLUE_GREEN_MAX = np.array([125, 110, 60], dtype=np.uint8) # 藍, 綠, 紅

# --- 新增：除錯圖片保存設定 ---
SAVE_DEBUG_IMAGES = True # <--- 這裡控制是否保存除錯圖片 (True/False)
DEBUG_IMAGES_FOLDER = "debug_board_scans" # <--- 圖片保存的資料夾名稱


# --- 輔助函數：判斷顏色是否在深藍綠色範圍內 ---
def is_deep_blue_green(bgr_color: np.ndarray) -> bool:
    """
    判斷給定的 BGR 顏色是否在預定義的深藍綠色背景範圍內。
    """
    return np.all(bgr_color >= DEEP_BLUE_GREEN_MIN) and np.all(bgr_color <= DEEP_BLUE_GREEN_MAX)

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
            x_end = x_start + sub_w

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
    # overall_mean = np.mean(avg_colors_np, axis=0) # 這裡實際上用的是9個子區域平均的平均值，不是整個圖片的平均

    if np.all((max_avg_color - min_avg_color) < tolerance):
        return True, np.mean(img_check, axis=(0, 1)), sub_region_avg_colors_list # 返回整個圖片的平均顏色
    
    return False, np.mean(img_check, axis=(0, 1)), sub_region_avg_colors_list # 同樣返回整個圖片的平均顏色

class LianLianKanBot:
    def __init__(self, template_folder="accumulated_templates_library"): # <-- 這裡已更新模板資料夾路徑
        self.template_folder = template_folder
        self.BOARD_ROWS = BOARD_ROWS
        self.BOARD_COLS = BOARD_COLS
        self.BOARD_TILE_WIDTH = TILE_WIDTH
        self.BOARD_TILE_HEIGHT = TILE_HEIGHT
        self.TILE_GAP_X = TILE_GAP_X
        self.TILE_GAP_Y = TILE_GAP_Y

        # 遊戲棋盤的偏移量 (從全螢幕左上角開始計算)
        self.GAME_BOARD_OFFSET_X = GAME_BOARD_OFFSET_X
        self.GAME_BOARD_OFFSET_Y = GAME_BOARD_OFFSET_Y
        self.GAME_BOARD_WIDTH = GAME_BOARD_WIDTH
        self.GAME_BOARD_HEIGHT = GAME_BOARD_HEIGHT

        # 裁剪的內部區域參數 (需要與模板生成器中的設置一致)
        self.INNER_TILE_PADDING = 8 # 普通模板內縮的像素 (這個值你上次設置正確了)
        self.PRE_PURE_COLOR_CHECK_PADDING = 5 # 用於純色判斷的區域從 tile_img_cropped 再次內縮的像素

        # 以下尺寸值應根據模板生成器的最終輸出進行設置
        self.NORMAL_TEMPLATE_WIDTH = 48 # <--- 修改為 48 (與你的模板一致)
        self.NORMAL_TEMPLATE_HEIGHT = 62 # <--- 修改為 62 (與你的模板一致)
        self.PURE_COLOR_CHECK_WIDTH = self.NORMAL_TEMPLATE_WIDTH - (2 * self.PRE_PURE_COLOR_CHECK_PADDING) # 48 - (2 * 5) = 38
        self.PURE_COLOR_CHECK_HEIGHT = self.NORMAL_TEMPLATE_HEIGHT - (2 * self.PRE_PURE_COLOR_CHECK_PADDING) # 62 - (2 * 5) = 52
        
        # 哈希閾值和純色判斷容忍度
        self.HASH_THRESHOLD = 15 # <--- 嘗試提高這個值，例如從 10 提高到 15 甚至 20，用於鎖定方塊
        self.EMPTY_TILE_COLOR_TOLERANCE = 15 # 用於純色判斷，應與模板生成器中的值一致

        # Load templates
        self.templates: Dict[str, Tuple[Image.Image, Any, np.ndarray]] = self._load_templates(template_folder)
        print(f"Loaded {len(self.templates)} general templates from {template_folder}")

        # 確保空模板和鎖定模板的哈希值被單獨儲存
        self.empty_template_hash: Optional[Any] = None
        self.locked_template_hashes: List[Any] = []

        empty_path = os.path.join(template_folder, "template_empty.png")
        if os.path.exists(empty_path):
            try:
                self.empty_template_hash = phash(Image.open(empty_path))
            except Exception as e:
                print(f"WARNING: Could not load empty template hash: {e}")

        # 加載所有鎖定模板 (假設名稱為 template_locked_1.png 到 template_locked_4.png)
        for i in range(1, 5): # <-- 這裡已更新為範圍 1 到 4，以匹配您說的 4 個鎖定模板
            locked_path = os.path.join(template_folder, f"template_locked_{i}.png")
            if os.path.exists(locked_path):
                try:
                    self.locked_template_hashes.append(phash(Image.open(locked_path)))
                except Exception as e:
                    print(f"WARNING: Could not load locked template hash {i}: {e}")
        print(f"Loaded {len(self.locked_template_hashes)} locked templates and 1 empty template.")


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
        返回 (board_matrix, tile_centers, debug_img)。
        """
        board_matrix = [["" for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
        tile_centers = {} # 儲存每個可點擊方塊的中心絕對座標 (相對於全螢幕)
        
        debug_img = game_window_img.copy()

        board_img = game_window_img[
            self.GAME_BOARD_OFFSET_Y : self.GAME_BOARD_OFFSET_Y + self.GAME_BOARD_HEIGHT,
            self.GAME_BOARD_OFFSET_X : self.GAME_BOARD_OFFSET_X + self.GAME_BOARD_WIDTH
        ]

        if board_img.size == 0:
            print("❌ 截取的棋盤區域為空，請檢查棋盤座標設定。")
            return board_matrix, tile_centers, debug_img

        # <--- 新增：確保 SAVE_DEBUG_IMAGES 和 DEBUG_IMAGES_FOLDER 可用
        # 我們需要從全局範圍訪問這些變數，雖然 _get_game_board_tiles 是方法，
        # 但Bot實例本身不直接持有這些配置，它們是全局配置。
        # 這裡不加 global 關鍵字，因為它們是全局模塊變量，方法可以讀取。
        # 但為了確保邏輯清晰，我們可以在需要時檢查它們。

        # 這裡增加一個內部計數器，用於每個方塊的命名
        tile_scan_count = 0 

        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                x = c * (self.BOARD_TILE_WIDTH + self.TILE_GAP_X)
                y = r * (self.BOARD_TILE_HEIGHT + self.TILE_GAP_Y)

                tile_img = board_img[y : y + self.BOARD_TILE_HEIGHT, x : x + self.BOARD_TILE_WIDTH]

                if tile_img.size == 0:
                    board_matrix[r][c] = "空ROI"
                    cv2.rectangle(debug_img,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (100, 100, 100), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img, "E_ROI", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (100, 100, 100), TEXT_THICKNESS)
                    continue
                
                cropped_y_start = self.INNER_TILE_PADDING
                cropped_y_end = self.INNER_TILE_PADDING + self.NORMAL_TEMPLATE_HEIGHT
                cropped_x_start = self.INNER_TILE_PADDING
                cropped_x_end = self.INNER_TILE_PADDING + self.NORMAL_TEMPLATE_WIDTH

                if not (0 <= cropped_y_start < cropped_y_end <= tile_img.shape[0] and
                        0 <= cropped_x_start < cropped_x_end <= tile_img.shape[1]):
                    board_matrix[r][c] = "裁剪錯誤"
                    cv2.rectangle(debug_img,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 0, 255), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img, "CROP_ERR", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (0, 0, 255), TEXT_THICKNESS)
                    continue

                tile_img_cropped = tile_img[
                    cropped_y_start : cropped_y_end,
                    cropped_x_start : cropped_x_end
                ]

                # <--- 新增：保存每個裁剪後的方塊圖片 ---
                # 需要在 main 函數的全局變數中檢查 SAVE_DEBUG_IMAGES
                # 但為了避免循環引用或在類中引入全局變量，我們可以在這裡重新判斷
                # 假設 `SAVE_DEBUG_IMAGES` 已經在 `main` 函數中被設置和處理
                # 最簡單的方法是直接引用它，因為它在模塊的頂層定義
                if globals()['SAVE_DEBUG_IMAGES']: # <--- 使用 globals() 訪問全局變數
                    tile_scan_count += 1
                    # 命名格式：scan_00001_row0_col0.png, scan_00001_row0_col1.png, ...
                    # 我們使用 main 函數中的全局 scan_count 來作為批次號
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
                # --- 結束新增 ---

                # --- 純色及深藍綠色背景判斷 ---
                pure_check_y_start = self.PRE_PURE_COLOR_CHECK_PADDING
                pure_check_y_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_HEIGHT
                pure_check_x_start = self.PRE_PURE_COLOR_CHECK_PADDING
                pure_check_x_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_WIDTH

                pure_color_check_area = None
                if (0 <= pure_check_y_start < pure_check_y_end <= tile_img_cropped.shape[0] and
                    0 <= pure_check_x_start < pure_check_x_end <= tile_img_cropped.shape[1]):
                    pure_color_check_area = tile_img_cropped[
                        pure_check_y_start : pure_check_y_end,
                        pure_check_x_start : pure_check_x_end
                    ]
                else:
                    pure_color_check_area = tile_img_cropped # 回退到使用整個裁剪區域

                if pure_color_check_area.size > 0:
                    is_nearly_solid, overall_mean_color, _ = is_nearly_solid_color(pure_color_check_area, self.EMPTY_TILE_COLOR_TOLERANCE)
                    
                    if is_nearly_solid and overall_mean_color is not None:
                        if is_deep_blue_green(overall_mean_color):
                            board_matrix[r][c] = "empty"
                            cv2.rectangle(debug_img,
                                          (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                          (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                          (0, 255, 0), RECTANGLE_THICKNESS)
                            cv2.putText(debug_img, "E", (self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                            self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1]) // 2), 
                                         cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
                            continue # 跳到下一個方塊

                # --- 哈希匹配邏輯 (如果不是深藍綠色背景，則執行以下代碼) ---
                if tile_img_cropped.size == 0:
                    board_matrix[r][c] = "裁剪錯誤"
                    cv2.rectangle(debug_img,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 0, 255), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img, "CROP_ERR", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (0, 0, 255), TEXT_THICKNESS)
                    continue

                pil_tile_img = Image.fromarray(cv2.cvtColor(tile_img_cropped, cv2.COLOR_BGR2RGB))
                current_tile_hash = phash(pil_tile_img)

                # 優先判斷是否為「空模板」
                if self.empty_template_hash and (current_tile_hash - self.empty_template_hash) <= self.HASH_THRESHOLD:
                    board_matrix[r][c] = "empty"
                    cv2.rectangle(debug_img,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 255, 0), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img, "E", (self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                            self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1]) // 2), 
                                         cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
                    continue

                # 判斷是否為「鎖定模板」
                is_locked_tile = False
                for i, locked_hash in enumerate(self.locked_template_hashes):
                    if (current_tile_hash - locked_hash) <= self.HASH_THRESHOLD:
                        board_matrix[r][c] = f"lock{i+1}"
                        cv2.rectangle(debug_img,
                                      (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                      (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                      (0, 0, 255), RECTANGLE_THICKNESS)
                        cv2.putText(debug_img, f"L{i+1}", (self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - cv2.getTextSize(f"L{i+1}", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                            self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + cv2.getTextSize(f"L{i+1}", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1]) // 2), 
                                             cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), TEXT_THICKNESS)
                        is_locked_tile = True
                        break
                if is_locked_tile:
                    continue

                # --- 普通模板匹配邏輯 (使用哈希匹配) ---
                matched_id = "未知"
                best_hash_distance = float('inf')

                for template_id_str, (_, template_hash_obj, _) in self.templates.items():
                    distance = current_tile_hash - template_hash_obj
                    if distance <= self.HASH_THRESHOLD and distance < best_hash_distance:
                        best_hash_distance = distance
                        matched_id = template_id_str
                
                board_matrix[r][c] = matched_id
                
                abs_x = self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH // 2
                abs_y = self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2
                tile_centers[(r, c)] = (abs_x, abs_y)

                if matched_id != "未知":
                    text_color = (255, 255, 0) # 青色 (B,G,R) for matched tiles
                    label = matched_id
                else:
                    text_color = (0, 255, 255) # 黃綠色 (B,G,R) for unknown tiles
                    label = "U"

                cv2.rectangle(debug_img,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (255, 0, 0), RECTANGLE_THICKNESS) # 藍色框
                
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
                text_pos_x = self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - text_size[0]) // 2
                text_pos_y = self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + text_size[1]) // 2
                cv2.putText(debug_img, label, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, TEXT_THICKNESS)

        return board_matrix, tile_centers, debug_img

    def is_passable(self, board, r, c, p1_coords, p2_coords):
        """
        檢查 (r, c) 座標是否可通行。
        可通行意味著它是空方塊，或者是當前正在檢查的 p1 或 p2 方塊。
        此函數也考慮了棋盤外部的無限空白區域。
        """
        # 處理棋盤外部的點
        if not (0 <= r < self.BOARD_ROWS and 0 <= c < self.BOARD_COLS):
            return True # 棋盤外部視為可通行

        # 處理棋盤內部的點
        current_coords = (r, c)
        if current_coords == p1_coords or current_coords == p2_coords:
            return True # 起點或終點自身可通行

        # 只有 "empty" 方塊才可通行
        return board[r][c] == "empty"

    def check_path_between(self, board, p_start, p_end, p1_original, p2_original):
        """
        檢查 p_start 到 p_end 之間是否有直線路徑，考慮 p1_original 和 p2_original 為可通過。
        """
        r_start, c_start = p_start
        r_end, c_end = p_end

        # 水平連線
        if r_start == r_end:
            step = 1 if c_end > c_start else -1
            for c in range(c_start + step, c_end, step):
                if not self.is_passable(board, r_start, c, p1_original, p2_original):
                    return False
            return True
        # 垂直連線
        elif c_start == c_end:
            step = 1 if r_end > r_start else -1
            for r in range(r_start + step, r_end, step):
                if not self.is_passable(board, r, c_start, p1_original, p2_original):
                    return False
            return True
        return False # 非直線

    def can_eliminate(self, board, p1, p2):
        r1, c1 = p1
        r2, c2 = p2

        if p1 == p2: return False

        tile1_name = board[r1][c1]
        tile2_name = board[r2][c2]

        # 不可點擊/不可連線的方塊類型
        non_playable_tiles = ["empty", "lock1", "lock2", "lock3", "lock4", "未知", "裁剪錯誤", "空ROI"]
        if tile1_name in non_playable_tiles or tile2_name in non_playable_tiles: return False

        # 只有相同類型的方塊才能消除
        if tile1_name != tile2_name: return False

        # --- 0 轉角 (直線) 連線 ---
        if self.check_path_between(board, p1, p2, p1, p2):
            return True

        # --- 1 轉角 (一個彎) 連線 ---
        # 嘗試 (r1, c2) 作為轉折點
        turn_point1 = (r1, c2)
        if self.is_passable(board, *turn_point1, p1, p2) and \
           self.check_path_between(board, p1, turn_point1, p1, p2) and \
           self.check_path_between(board, turn_point1, p2, p1, p2):
            return True

        # 嘗試 (r2, c1) 作為轉折點
        turn_point2 = (r2, c1)
        if self.is_passable(board, *turn_point2, p1, p2) and \
           self.check_path_between(board, p1, turn_point2, p1, p2) and \
           self.check_path_between(board, turn_point2, p2, p1, p2):
            return True

        # --- 2 轉角 (兩個彎) 連線 ---
        # 遍歷所有可能的中間點 (包括棋盤外部擴展一格)
        # 這裡的範圍從 -1 到 self.BOARD_ROWS/COLS + 1 是為了模擬棋盤外的「無限」空區域
        for r_mid in range(-1, self.BOARD_ROWS + 1):
            for c_mid in range(-1, self.BOARD_COLS + 1):
                mid_point = (r_mid, c_mid)

                # 如果中間點是p1或p2本身，則跳過，因為這會被0或1轉角處理
                if mid_point == p1 or mid_point == p2:
                    continue
                
                # 如果中間點不是空方塊且在棋盤內，則不能作為中轉點
                # is_passable 已經處理了棋盤外的點為 True 的情況
                if not self.is_passable(board, *mid_point, p1, p2):
                    continue

                # 檢查 p1 -> mid_point 和 mid_point -> p2 是否都是直線路徑
                if self.check_path_between(board, p1, mid_point, p1, p2) and \
                   self.check_path_between(board, mid_point, p2, p1, p2):
                    
                    # 判斷是否為兩轉角：路徑不是直線，且中點不是 p1 或 p2
                    # 判斷標準：P1-mid 和 mid-P2 必須是直角連接，且整體不是直線。
                    # 如果 p1, mid_point, p2 都在同一直線 (水平或垂直)，則屬於 0 轉角，不應被 2 轉角捕捉
                    # 如果 p1, mid_point, p2 構成一個 L 形 (一個轉彎)，則屬於 1 轉角，不應被 2 轉角捕捉
                    
                    # 排除直線情況 (0轉角)
                    if (r1 == r2 and r1 == r_mid) or (c1 == c2 and c1 == c_mid): # P1, P2, MidPoint 在同一直線
                        continue
                    
                    # 排除一個轉彎情況 (1轉角)
                    # 如果 mid_point 與 p1 或 p2 的其中一個在同一行/列，且與另一個在同一列/行，且 mid_point 位於 p1 和 p2 的 L 型轉角處
                    if (r1 == r_mid and c2 == c_mid) or (r2 == r_mid and c1 == c_mid):
                        continue

                    # 只要通過了上述排除，且滿足兩段直線，就認為是兩轉角
                    return True
        return False

def get_window_by_title(title: str):
    """根據標題尋找視窗句柄"""
    hwnd = win32gui.FindWindow(None, title)
    return hwnd

# --- 全域變數 for 視窗資訊和腳本狀態 ---
game_window_x, game_window_y = 0, 0
running = False
display_window_name = "Lianliankan Bot Debug"
last_display_img = None
bot_instance: Optional[LianLianKanBot] = None
scan_count = 0 # <--- 新增：記錄掃描次數，用於檔案命名

# --- 熱鍵回調函數 ---
def start_bot():
    global running
    running = True
    print("\n--- 外掛已啟動 (按 F4 暫停, Q 鍵退出) ---")

def stop_bot():
    global running
    running = False
    print("\n--- 外掛已暫停 (按 F3 啟動, Q 鍵退出) ---")

def exit_program():
    global running
    running = False
    sys.exit(0)

# <--- 新增：開關除錯圖片保存的熱鍵函數 ---
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
        # 同時清空 tiles 子資料夾
        tiles_folder = os.path.join(DEBUG_IMAGES_FOLDER, "tiles")
        if os.path.exists(tiles_folder):
            for f in os.listdir(tiles_folder):
                try:
                    os.remove(os.path.join(tiles_folder, f))
                except OSError as e:
                    print(f"錯誤: 無法刪除檔案 {f} - {e}")
            try:
                os.rmdir(tiles_folder) # 移除資料夾本身
            except OSError as e:
                print(f"錯誤: 無法刪除資料夾 {tiles_folder} - {e}")
        
        # 重新創建 tiles 資料夾 (確保存在)
        if not os.path.exists(tiles_folder):
            os.makedirs(tiles_folder)

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
    # <--- 新增這裡：確保 main 函數也能正確訪問和更新 SAVE_DEBUG_IMAGES
    global SAVE_DEBUG_IMAGES, DEBUG_IMAGES_FOLDER 

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

    # 初始化 Bot 實例，傳遞新的模板資料夾名稱
    bot_instance = LianLianKanBot(template_folder="accumulated_templates_library") 

    # <--- 新增這裡：Bot 啟動時立即檢查 SAVE_DEBUG_IMAGES 的狀態並執行相應操作
    if SAVE_DEBUG_IMAGES:
        # 確保資料夾存在
        if not os.path.exists(DEBUG_IMAGES_FOLDER):
            os.makedirs(DEBUG_IMAGES_FOLDER)
        print(f"--- 除錯圖片保存已 開啟 ---") # 新增這行輸出
        print(f"清空舊的除錯圖片在 {DEBUG_IMAGES_FOLDER}...")
        
        # 清空 tiles 子資料夾
        tiles_folder = os.path.join(DEBUG_IMAGES_FOLDER, "tiles")
        if os.path.exists(tiles_folder):
            for f in os.listdir(tiles_folder):
                try:
                    os.remove(os.path.join(tiles_folder, f))
                except OSError as e:
                    print(f"錯誤: 無法刪除檔案 {f} - {e}")
            try:
                os.rmdir(tiles_folder) # 移除資料夾本身
            except OSError as e:
                print(f"錯誤: 無法刪除資料夾 {tiles_folder} - {e}")
        
        # 重新創建 tiles 資料夾 (確保存在)
        if not os.path.exists(tiles_folder):
            os.makedirs(tiles_folder)

        # 清空主資料夾裡的 .png 圖片
        for f in os.listdir(DEBUG_IMAGES_FOLDER):
            if f.endswith(".png"): # 只刪除 .png 文件
                try:
                    os.remove(os.path.join(DEBUG_IMAGES_FOLDER, f))
                except OSError as e:
                    print(f"錯誤: 無法刪除檔案 {f} - {e}")
        print("舊圖片已清空。")
    else:
        print(f"--- 除錯圖片保存已 關閉 ---") # 新增這行輸出
    # --- 結束新增 ---

    # 創建用於顯示的 OpenCV 視窗
    cv2.namedWindow(display_window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(display_window_name, cv2.WND_PROP_TOPMOST, 1)

    # 調整 Debug 視窗大小和位置
    display_width = 1200
    display_height = 700
    display_pos_x = 0
    display_pos_y = 0
    cv2.resizeWindow(display_window_name, display_width, display_height)
    cv2.moveWindow(display_window_name, display_pos_x, display_pos_y)

    while True:
        if running:
            try:
                rect = win32gui.GetWindowRect(hwnd)
                game_window_x, game_window_y, x2, y2 = rect
                window_width, window_height = x2 - game_window_x, y2 - game_window_y

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
                
                board_matrix, tile_centers, display_img = bot_instance._get_game_board_tiles(img_game_window)

                # <--- 新增：保存除錯圖片邏輯 ---
                if SAVE_DEBUG_IMAGES:
                    scan_count += 1
                    save_path = os.path.join(DEBUG_IMAGES_FOLDER, f"scan_{scan_count:05d}_board_with_labels.png")
                    try:
                        cv2.imwrite(save_path, display_img)
                        # print(f"已保存除錯圖片: {save_path}") # 可以取消註釋這行，如果你想看每次保存的提示
                    except Exception as e:
                        print(f"❌ 保存除錯圖片失敗: {e}")
                # --- 結束新增 ---

                found_pair_and_clicked = False
                for r1 in range(BOARD_ROWS):
                    for c1 in range(BOARD_COLS):
                        p1 = (r1, c1)
                        non_clickable_check = ["empty", "lock1", "lock2", "lock3", "lock4", "未知", "裁剪錯誤", "空ROI"] # 增加了lock3, lock4
                        if board_matrix[r1][c1] in non_clickable_check or p1 not in tile_centers:
                            continue

                        for r2 in range(BOARD_ROWS):
                            for c2 in range(BOARD_COLS):
                                p2 = (r2, c2)
                                if p2 not in tile_centers or p1 == p2:
                                    continue

                                if board_matrix[r1][c1] == board_matrix[r2][c2]:
                                    if bot_instance.can_eliminate(board_matrix, p1, p2):
                                        print(f"✅ 找到可消除的對子: {board_matrix[r1][c1]} 在 {p1} 和 {p2}")

                                        draw_x1, draw_y1 = tile_centers[p1]
                                        draw_x2, draw_y2 = tile_centers[p2]
                                        
                                        # 轉換為相對於 Debug 視窗的座標進行繪圖
                                        cv2.circle(display_img, (draw_x1 - game_window_x, draw_y1 - game_window_y), 10, LINE_COLOR, -1)
                                        cv2.circle(display_img, (draw_x2 - game_window_x, draw_y2 - game_window_y), 10, LINE_COLOR, -1)
                                        cv2.line(display_img, (draw_x1 - game_window_x, draw_y1 - game_window_y), (draw_x2 - game_window_x, draw_y2 - game_window_y), LINE_COLOR, LINE_THICKNESS)
                                        
                                        pyautogui.click(draw_x1, draw_y1)
                                        time.sleep(CLICK_DELAY)
                                        pyautogui.click(draw_x2, draw_y2)
                                        time.sleep(CLICK_DELAY)

                                        found_pair_and_clicked = True
                                        break
                            if found_pair_and_clicked:
                                break
                        if found_pair_and_clicked:
                            break

                display_img_resized = cv2.resize(display_img, (display_width, display_height))
                cv2.imshow(display_window_name, display_img_resized)

                last_display_img = display_img.copy()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("檢測到 Q 鍵，程式即將退出。")
                    break

                time.sleep(SCAN_INTERVAL)

            except Exception as e:
                print(f"運行過程中發生錯誤: {e}")
                print("將在 5 秒後重試...")
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
                print("檢測到 Q 鍵，程式即將退出。")
                break
            time.sleep(0.5)

    cv2.destroyAllWindows()
    keyboard.unhook_all()

if __name__ == "__main__":
    main()