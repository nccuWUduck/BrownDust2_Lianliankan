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

# --- 除錯圖片保存設定 ---
SAVE_DEBUG_IMAGES = False # <--- 這裡控制是否保存除錯圖片 (True/False) - 默認修改為 False
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
        self.TILE_GAP_X = TILE_GAP_X
        self.TILE_GAP_Y = TILE_GAP_Y

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
        self.HASH_THRESHOLD = 15 # 普通圖標和鎖定圖標的哈希閾值
        self.EMPTY_TILE_HASH_THRESHOLD = 18 # 空方塊專屬的哈希閾值，可以更高，因為背景可能變化更大
        self.EMPTY_TILE_COLOR_TOLERANCE = 25 # 純色判斷容忍度，以適應更多背景變化

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
                x = c * (self.BOARD_TILE_WIDTH + self.TILE_GAP_X)
                y = r * (self.BOARD_TILE_HEIGHT + self.TILE_GAP_Y)

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
                # --- 結束保存 ---

                if tile_img_cropped.size == 0:
                    board_matrix[r][c] = "裁剪錯誤"
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 0, 255), RECTANGLE_THICKNESS)
                    cv2.putText(debug_img_base, "CROP_ERR", (self.GAME_BOARD_OFFSET_X + x + 5, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE * 0.7, (0, 0, 255), TEXT_THICKNESS)
                    continue

                pil_tile_img = Image.fromarray(cv2.cvtColor(tile_img_cropped, cv2.COLOR_BGR2RGB))
                current_tile_hash = phash(pil_tile_img)

                matched_id = "未知" # 默認為未知
                
                # --- 1. 優先判斷是否為「鎖定模板」 ---
                for i, locked_hash in enumerate(self.locked_template_hashes):
                    if (current_tile_hash - locked_hash) <= self.HASH_THRESHOLD:
                        matched_id = f"lock{i+1}"
                        break
                
                if matched_id != "未知":
                    board_matrix[r][c] = matched_id
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 0, 255), RECTANGLE_THICKNESS) # 紅色框表示鎖定
                    cv2.putText(debug_img_base, f"L{i+1}", (self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - cv2.getTextSize(f"L{i+1}", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                        self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + cv2.getTextSize(f"L{i+1}", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1]) // 2), 
                                         cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), TEXT_THICKNESS)
                    abs_x = self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH // 2
                    abs_y = self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2
                    tile_centers[(r, c)] = (abs_x, abs_y) # 鎖定方塊也計算中心點，雖然不可點擊但能用於路徑判斷
                    continue 

                # --- 2. 其次判斷是否為「普通模板」 ---
                best_hash_distance = float('inf')
                for template_id_str, (_, template_hash_obj, _) in self.templates.items():
                    distance = current_tile_hash - template_hash_obj
                    if distance <= self.HASH_THRESHOLD and distance < best_hash_distance:
                        best_hash_distance = distance
                        matched_id = template_id_str
                
                if matched_id != "未知":
                    board_matrix[r][c] = matched_id
                    abs_x = self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH // 2
                    abs_y = self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT // 2
                    tile_centers[(r, c)] = (abs_x, abs_y)
                    text_color = (255, 255, 0) # 青色 (B,G,R) for matched tiles
                    label = matched_id
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (255, 0, 0), RECTANGLE_THICKNESS) # 藍色框
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
                    text_pos_x = self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - text_size[0]) // 2
                    text_pos_y = self.GAME_BOARD_OFFSET_Y + y + (text_size[1] + 10) 
                    cv2.putText(debug_img_base, label, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, TEXT_THICKNESS)
                    continue

                # --- 3. 最後判斷是否為「空方塊」 (只有在未匹配到任何圖標時才執行) ---
                is_empty_tile = False
                
                # 將純色檢查區域的定義放在這裡，確保其總是能被訪問到
                pure_check_y_start = self.PRE_PURE_COLOR_CHECK_PADDING
                pure_check_y_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_HEIGHT
                pure_check_x_start = self.PRE_PURE_COLOR_CHECK_PADDING
                pure_check_x_end = self.PRE_PURE_COLOR_CHECK_PADDING + self.PURE_COLOR_CHECK_WIDTH

                pure_color_check_area = None
                # 檢查裁剪範圍是否有效，如果無效則回退到使用整個 tile_img_cropped
                if (0 <= pure_check_y_start < pure_check_y_end <= tile_img_cropped.shape[0] and
                    0 <= pure_check_x_start < pure_check_x_end <= tile_img_cropped.shape[1]):
                    pure_color_check_area = tile_img_cropped[
                        pure_check_y_start : pure_check_y_end,
                        pure_check_x_start : pure_check_x_end
                    ]
                else:
                    # 如果內縮裁剪無效，則使用整個 tile_img_cropped 進行純色判斷
                    # 這可能是由於 TILE_WIDTH/HEIGHT 太小導致的邊界情況
                    print(f"⚠️ 警告: 純色檢查區域裁剪無效，將使用整個方塊圖片 ({r},{c})。")
                    pure_color_check_area = tile_img_cropped 

                # 3.1. 嘗試使用純色深藍綠色判斷空方塊 (基於舊背景)
                if pure_color_check_area.size > 0:
                    is_nearly_solid, overall_mean_color, _ = is_nearly_solid_color(pure_color_check_area, self.EMPTY_TILE_COLOR_TOLERANCE)
                    if is_nearly_solid and overall_mean_color is not None and is_deep_blue_green(overall_mean_color):
                        is_empty_tile = True

                # 3.2. 如果不是藍綠色純色，嘗試使用紅色條紋空模板哈希匹配 (基於新背景)
                if not is_empty_tile and self.empty_red_striped_template_hash and \
                   (current_tile_hash - self.empty_red_striped_template_hash) <= self.EMPTY_TILE_HASH_THRESHOLD:
                    is_empty_tile = True
                
                # 3.3. 最後，如果藍綠色純色判斷未啟用或未匹配，並且有藍綠色空模板，嘗試使用哈希匹配作為備用
                if not is_empty_tile and self.empty_blue_green_template_hash and \
                   (current_tile_hash - self.empty_blue_green_template_hash) <= self.EMPTY_TILE_HASH_THRESHOLD:
                    is_empty_tile = True


                if is_empty_tile:
                    board_matrix[r][c] = "empty"
                    cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 255, 0), RECTANGLE_THICKNESS) # 綠色框表示空
                    cv2.putText(debug_img_base, "E", (self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                    self.GAME_BOARD_OFFSET_Y + y + (self.BOARD_TILE_HEIGHT + cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1]) // 2), 
                                 cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
                    continue 
                
                # --- 如果以上都未匹配到，則為「未知」方塊 ---
                board_matrix[r][c] = "未知"
                text_color = (0, 255, 255) # 黃色 (B,G,R) for unknown tiles
                label = "U"

                cv2.rectangle(debug_img_base,
                                  (self.GAME_BOARD_OFFSET_X + x, self.GAME_BOARD_OFFSET_Y + y),
                                  (self.GAME_BOARD_OFFSET_X + x + self.BOARD_TILE_WIDTH, self.GAME_BOARD_OFFSET_Y + y + self.BOARD_TILE_HEIGHT),
                                  (0, 165, 255), RECTANGLE_THICKNESS) # 橙色框 (B,G,R) for unknown
                
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)
                text_pos_x = self.GAME_BOARD_OFFSET_X + x + (self.BOARD_TILE_WIDTH - text_size[0]) // 2
                text_pos_y = self.GAME_BOARD_OFFSET_Y + y + (text_size[1] + 10) 
                cv2.putText(debug_img_base, label, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, TEXT_THICKNESS)
                
                # 如果是未知方塊，保存其裁剪後的圖片以供進一步分析
                if globals()['SAVE_DEBUG_IMAGES']:
                    unknown_tile_save_name = f"unknown_{globals()['scan_count']:05d}_r{r:02d}_c{c:02d}.png"
                    unknown_tile_save_path = os.path.join(unknown_tiles_folder, unknown_tile_save_name)
                    try:
                        cv2.imwrite(unknown_tile_save_path, tile_img_cropped)
                        # print(f"已保存未知方塊圖片: {unknown_tile_save_path}") # 這行會印很多，暫時註釋
                    except Exception as e:
                        print(f"❌ 保存未知方塊圖片失敗: {unknown_tile_save_path} - {e}")


        return board_matrix, tile_centers, debug_img_base

    def is_passable(self, board, r, c, p1_coords, p2_coords):
        """
        檢查 (r, c) 座標是否可通行。
        可通行意味著它是空方塊，或者是當前正在檢查的 p1 或 p2 方塊。
        此函數也考慮了棋盤外部的無限空白區域。
        """
        # 處理棋盤外部的點：任何在棋盤邊界外的點都被視為「可通行」的虛擬空區域。
        if not (0 <= r < self.BOARD_ROWS and 0 <= c < self.BOARD_COLS):
            return True 

        # 處理棋盤內部的點
        current_coords = (r, c)
        # 起點或終點自身總是可通行，因為它們是我們嘗試連接的方塊。
        if current_coords == p1_coords or current_coords == p2_coords:
            return True 

        # 在棋盤內部，只有 "empty" 方塊才可通行
        return board[r][c] == "empty"

    def check_path_between(self, board, p_start, p_end, p1_original, p2_original):
        """
        檢查 p_start 到 p_end 之間是否有直線路徑。
        路徑中間的方塊必須是可通行 (empty 或 p1_original/p2_original)。
        這個函數也能正確處理起點或終點在棋盤外的情況。
        """
        r_start, c_start = p_start
        r_end, c_end = p_end

        # 水平連線
        if r_start == r_end:
            step = 1 if c_end > c_start else -1
            # 遍歷從起點到終點之間的所有中間點
            for c_check in range(c_start + step, c_end, step):
                if not self.is_passable(board, r_start, c_check, p1_original, p2_original):
                    return False # 遇到障礙
            return True # 成功找到直線路徑
        # 垂直連線
        elif c_start == c_end:
            step = 1 if r_end > r_start else -1
            # 遍歷從起點到終點之間的所有中間點
            for r_check in range(r_start + step, r_end, step):
                if not self.is_passable(board, r_check, c_start, p1_original, p2_original):
                    return False # 遇到障礙
            return True # 成功找到直線路徑
        return False # 非直線連線

    def can_eliminate(self, board, p1, p2):
        r1, c1 = p1
        r2, c2 = p2

        if p1 == p2: return False

        tile1_name = board[r1][c1]
        tile2_name = board[r2][c2]

        # 不可點擊/不可連線的方塊類型 (包括空白、鎖定、未知或錯誤的方塊)
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
        # 這種情況的連線通常會延伸到棋盤外圍的「虛擬空區域」。
        # 我們遍歷所有可能的「中間轉折點 M」，這些點可以在棋盤內部，也可以在棋盤外圍一圈。
        # 循環範圍 `range(-1, self.BOARD_ROWS + 1)` 和 `range(-1, self.BOARD_COLS + 1)` 
        # 精確地實現了「多加一圈0」來模擬棋盤外圍的虛擬空區域。

        # 情況 1: P1 -> (水平線) -> 中間點 M1 -> (垂直線) -> 中間點 M2 -> (水平線) -> P2
        # (r1, c1) --H--> (r1, temp_c) --V--> (r2, temp_c) --H--> (r2, c2)
        for temp_c in range(-1, self.BOARD_COLS + 1): # 遍歷所有可能的中間列
            m1_point = (r1, temp_c) # 第一個轉折點 (與 P1 同行)
            m2_point = (r2, temp_c) # 第二個轉折點 (與 P2 同行，與 M1 同列)
            
            # 檢查三段直線路徑是否都可通行
            if self.check_path_between(board, p1, m1_point, p1, p2) and \
               self.check_path_between(board, m1_point, m2_point, p1, p2) and \
               self.check_path_between(board, m2_point, p2, p1, p2):
                return True

        # 情況 2: P1 -> (垂直線) -> 中間點 M1 -> (水平線) -> 中間點 M2 -> (垂直線) -> P2
        # (r1, c1) --V--> (temp_r, c1) --H--> (temp_r, c2) --V--> (r2, c2)
        for temp_r in range(-1, self.BOARD_ROWS + 1): # 遍歷所有可能的中間行
            m1_point = (temp_r, c1) # 第一個轉折點 (與 P1 同列)
            m2_point = (temp_r, c2) # 第二個轉折點 (與 P2 同列，與 M1 同行)
            
            # 檢查三段直線路徑是否都可通行
            if self.check_path_between(board, p1, m1_point, p1, p2) and \
               self.check_path_between(board, m1_point, m2_point, p1, p2) and \
               self.check_path_between(board, m2_point, p2, p1, p2):
                return True
        
        return False

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
    display_pos_x = 0
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

                # 1. 將本輪的 current_raw_board_matrix 存為上一輪的 last_raw_board_matrix (在獲取新掃描結果之前)
                last_raw_board_matrix = [row[:] for row in current_raw_board_matrix] 

                # 2. 獲取當前最新的原始棋盤掃描結果 (包含原始識別的 "U")
                current_raw_board_matrix, tile_centers, display_img = bot_instance._get_game_board_tiles(img_game_window)
                # 直接使用 debug_img_base 作為 display_img，後續在這個圖片上進行繪製

                # 3. 根據 last_raw_board_matrix 和 current_raw_board_matrix 更新 persisted_e_board
                #    同時在 display_img 上繪製 E 框
                non_solid_states = ["empty", "未知", "裁剪錯誤", "空ROI"] # 這些狀態代表不是一個「實體」方塊

                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        last_state = last_raw_board_matrix[r][c]
                        current_state = current_raw_board_matrix[r][c]

                        # 判斷是否為「實體」方塊 (非空、非錯誤、非未知)
                        is_last_solid = last_state not in non_solid_states
                        is_current_non_solid = current_state in non_solid_states

                        if is_last_solid and is_current_non_solid:
                            # 如果上一輪是實體方塊，這一輪變成非實體方塊 (空、U、錯誤等)
                            # 則標記為 E 狀態
                            persisted_e_board[r][c] = "E"
                        elif current_state not in non_solid_states:
                            # 如果當前狀態是實際的方塊 (被新方塊填充了)
                            # 清除該位置的 E 狀態
                            persisted_e_board[r][c] = ""
                        # 否則，保持 persisted_e_board[r][c] 不變 (例如：E 仍然是 E，空仍然是空，U仍然是U)
                        
                # 4. 創建 board_to_process, 初始是 current_raw_board_matrix 的副本
                board_to_process = [row[:] for row in current_raw_board_matrix] 

                # 5. 將 persisted_e_board 的 E 狀態應用到 board_to_process 和 display_img
                for r in range(BOARD_ROWS):
                    for c in range(BOARD_COLS):
                        if persisted_e_board[r][c] == "E":
                            board_to_process[r][c] = "empty" # 邏輯上視為 empty

                            # 在 display_img 上繪製綠色的 E 框 (覆蓋原始識別結果)
                            x_draw = c * (bot_instance.BOARD_TILE_WIDTH + bot_instance.TILE_GAP_X)
                            y_draw = r * (bot_instance.BOARD_TILE_HEIGHT + bot_instance.TILE_GAP_Y)
                            cv2.rectangle(display_img,
                                          (bot_instance.GAME_BOARD_OFFSET_X + x_draw, bot_instance.GAME_BOARD_OFFSET_Y + y_draw),
                                          (bot_instance.GAME_BOARD_OFFSET_X + x_draw + bot_instance.BOARD_TILE_WIDTH, bot_instance.GAME_BOARD_OFFSET_Y + y_draw + bot_instance.BOARD_TILE_HEIGHT),
                                          (0, 255, 0), RECTANGLE_THICKNESS) # 綠色框表示空
                            cv2.putText(display_img, "E", (bot_instance.GAME_BOARD_OFFSET_X + x_draw + (bot_instance.BOARD_TILE_WIDTH - cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][0]) // 2, 
                                                            bot_instance.GAME_BOARD_OFFSET_Y + y_draw + (cv2.getTextSize("E", cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_THICKNESS)[0][1] + 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), TEXT_THICKNESS)
                                        
                # --- END: 棋盤狀態管理邏輯 ---


                # --- 保存除錯圖片邏輯 ---
                if SAVE_DEBUG_IMAGES:
                    save_path = os.path.join(DEBUG_IMAGES_FOLDER, f"scan_{scan_count:05d}_board_with_labels.png")
                    try:
                        cv2.imwrite(save_path, display_img)
                        # print(f"已保存除錯圖片: {save_path}") # 可以取消註釋這行，如果你想看每次保存的提示
                    except Exception as e:
                        print(f"❌ 保存除錯圖片失敗: {e}")
                # --- 結束保存 ---

                # --- 尋找所有可消除的對子 ---
                eliminable_pairs = [] # 儲存所有可消除的 (p1, p2) 對
                
                # 使用 set 來記錄已經被加入 eliminable_pairs 的方塊，避免重複檢查
                processed_pairs = set() 

                for r1 in range(BOARD_ROWS):
                    for c1 in range(BOARD_COLS):
                        p1 = (r1, c1)
                        # 這裡使用 board_to_process 來獲取方塊類型 (已包含 E 修正)
                        non_clickable_check = ["empty", "lock1", "lock2", "lock3", "lock4", "未知", "裁剪錯誤", "空ROI"]
                        if board_to_process[r1][c1] in non_clickable_check or p1 not in tile_centers:
                            continue

                        for r2 in range(BOARD_ROWS):
                            for c2 in range(BOARD_COLS):
                                p2 = (r2, c2)
                                if p1 == p2 or p2 not in tile_centers:
                                    continue

                                # 確保不會重複處理同一對 (p1, p2) 和 (p2, p1)
                                if tuple(sorted((p1, p2))) in processed_pairs:
                                    continue

                                # 這裡使用 board_to_process 來判斷方塊是否相同
                                if board_to_process[r1][c1] == board_to_process[r2][c2]:
                                    # 檢查這個對子是否真的可以消除
                                    if bot_instance.can_eliminate(board_to_process, p1, p2): # 傳遞更新後的矩陣
                                        eliminable_pairs.append((p1, p2))
                                        processed_pairs.add(tuple(sorted((p1, p2)))) # 記錄這對已經處理過

                # --- 一次性點擊所有找到的對子 ---
                if eliminable_pairs:
                    print(f"本輪找到 {len(eliminable_pairs)} 對可消除的方塊，準備點擊...")

                    for p1, p2 in eliminable_pairs:
                        print(f"✅ 消除對子: {board_to_process[p1[0]][p1[1]]} 在 {p1} 和 {p2}")

                        if p1 in tile_centers and p2 in tile_centers:
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
                            
                            # 這裡不需要將座標加入到 recently_clicked_coords 了
                            # 因為 E 的判斷邏輯已經改為比較兩次掃描結果
                        else:
                            print(f"⚠️ 警告: 嘗試點擊的方塊 {p1} 或 {p2} 不在 tile_centers 中，跳過此對。")

                    # 點擊完成後，可能需要短暫延遲讓遊戲畫面刷新
                    time.sleep(0.5) 
                else:
                    print("本輪未找到可消除的對子。")
                    
                    # 這一部分邏輯簡化：不再主動清除 persisted_e_board，它會根據實際掃描結果自動清除
                    # 如果一個 E 位置被新的方塊覆蓋，persisted_e_board[r][c] 會在步驟 3 中被清除
                    # 如果 E 狀態持續存在 (表示它仍然是空的或 U)，那就讓它繼續存在，直到被實際方塊填充
                    
                display_img_resized = cv2.resize(display_img, (display_width, display_height))
                cv2.imshow(display_window_name, display_img_resized)

                last_display_img = display_img.copy()

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("檢測到 Q 鍵，程式即將退出。")
                    break

                time.sleep(SCAN_INTERVAL) # 每次掃描間隔

            except Exception as e:
                print(f"運行過程中發生錯誤: {e}")
                import traceback
                traceback.print_exc() # 打印完整的錯誤堆棧，便於調試
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