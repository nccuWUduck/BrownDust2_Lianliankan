import cv2
import numpy as np
import os

# --- 配置參數 ---
SOURCE_IMAGE_PATH = "game_window_screenshot_1.png" # 你的原始遊戲截圖路徑
OUTPUT_TEMPLATES_DIR = "templates" # 模板圖片保存的資料夾

# !!! 在這裡自訂 template_counter 的初始值 !!!
STARTING_TEMPLATE_COUNTER = 29 # <--- 將這裡的值修改為你想要的起始數字

# --- 全域變數用於存儲點擊點和狀態 ---
clicks = [] # 儲存點擊的座標 (x, y)
current_img = None # 儲存當前顯示的圖片 (會被繪製)
original_img = None # 儲存原始載入的圖片 (不被修改)
template_counter = STARTING_TEMPLATE_COUNTER # 使用自訂的初始值

# --- 滑鼠事件回調函數 ---
def mouse_callback(event, x, y, flags, param):
    global clicks, current_img, original_img, template_counter

    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"點擊座標: ({x}, {y}) - 已收集 {len(clicks)}/4 個點")

        # 在當前圖片上繪製點
        cv2.circle(current_img, (x, y), 5, (0, 255, 0), -1) # 綠色實心圓
        cv2.imshow("Image Cutter - Click 4 Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right)", current_img)

        if len(clicks) == 4:
            print("\n已收集 4 個點。正在處理模板裁剪...")
            process_points_and_crop()
            
            # 重置點擊，準備下一個方塊
            clicks = []
            # 重新載入原始圖片，以便進行新的裁剪
            current_img = original_img.copy()
            cv2.imshow("Image Cutter - Click 4 Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right)", current_img)
            print("\n請點擊下一個模板的四個點 (左上, 右上, 左下, 右下)。按 'q' 退出。")


def process_points_and_crop():
    global clicks, original_img, template_counter

    if len(clicks) != 4:
        print("錯誤: 需要 4 個點才能裁剪。")
        return

    # 從四個點中找出最小和最大的 X, Y 座標
    x_coords = [p[0] for p in clicks]
    y_coords = [p[1] for p in clicks]

    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # 計算裁剪區域的寬高
    width = max_x - min_x
    height = max_y - min_y

    # 確保裁剪區域有效
    if width <= 0 or height <= 0:
        print(f"錯誤: 裁剪區域無效 (寬度: {width}, 高度: {height})。請確保點擊的點能形成有效矩形。")
        return

    # 確保裁剪區域在圖片範圍內
    if min_x < 0 or min_y < 0 or max_x > original_img.shape[1] or max_y > original_img.shape[0]:
        print(f"錯誤: 裁剪區域超出原始圖片邊界。請重新點擊。")
        return

    # 裁剪圖片
    cropped_img = original_img[min_y:max_y, min_x:max_x]

    # 創建輸出目錄如果它不存在
    if not os.path.exists(OUTPUT_TEMPLATES_DIR):
        os.makedirs(OUTPUT_TEMPLATES_DIR)
        print(f"已創建輸出資料夾: {OUTPUT_TEMPLATES_DIR}")

    # 生成模板檔名
    template_name = f"template_{template_counter:03d}.png" # 例如: template_001.png, template_002.png
    output_path = os.path.join(OUTPUT_TEMPLATES_DIR, template_name)

    # 保存裁剪後的圖片
    cv2.imwrite(output_path, cropped_img)
    print(f"✨ 已保存模板: {output_path}")
    print(f"裁剪區域 (左上角X, Y, 寬度, 高度): ({min_x}, {min_y}, {width}, {height})")
    template_counter += 1


def run_interactive_cutter():
    global current_img, original_img

    # 檢查源截圖是否存在
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"❌ 錯誤: 源截圖文件 '{SOURCE_IMAGE_PATH}' 不存在。")
        print("請確保你已經抓取了遊戲截圖並將其放在正確的路徑下。")
        return

    # 載入原始截圖
    original_img = cv2.imread(SOURCE_IMAGE_PATH)
    if original_img is None:
        print(f"❌ 錯誤: 無法載入圖片 '{SOURCE_IMAGE_PATH}'。請檢查文件是否損壞或路徑是否正確。")
        return

    current_img = original_img.copy()

    cv2.namedWindow("Image Cutter - Click 4 Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")
    cv2.setMouseCallback("Image Cutter - Click 4 Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right)", mouse_callback)

    print("請點擊要裁剪的模板的四個點，點擊順序建議為：")
    print("1. 左上角")
    print("2. 右上角")
    print("3. 左下角")
    print("4. 右下角")
    print(f"模板將從編號 {STARTING_TEMPLATE_COUNTER} 開始保存。按 'q' 鍵退出程式。")

    while True:
        cv2.imshow("Image Cutter - Click 4 Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right)", current_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("\n程式已退出。")

if __name__ == "__main__":
    run_interactive_cutter()