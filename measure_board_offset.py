import pyautogui
import win32gui
import time
import os
import sys
import keyboard # 引入 keyboard 模組
import winsound # 引入 winsound 模組用於播放聲音

# --- 配置參數 ---
TARGET_WINDOW_TITLE = "BrownDust II" # 你的遊戲視窗標題

# --- 全域變數 ---
game_window_x, game_window_y = 0, 0 # 遊戲視窗在螢幕上的左上角絕對座標
game_hwnd = None # 遊戲視窗句柄
enter_pressed = False # 標誌 Enter 鍵是否被按下

# --- 輔助函數 ---

def get_window_by_title(title_keyword):
    """根據標題關鍵字獲取視窗句柄。"""
    hwnd = None
    def enum_windows_callback(hwnd_temp, extra):
        # 檢查視窗是否可見且標題包含關鍵字
        if win32gui.IsWindowVisible(hwnd_temp) and title_keyword.lower() in win32gui.GetWindowText(hwnd_temp).lower():
            extra.append(hwnd_temp)
    hwnds = []
    win32gui.EnumWindows(enum_windows_callback, hwnds)
    return hwnds[0] if hwnds else None

def get_game_window_coords():
    """更新遊戲視窗的螢幕絕對座標。"""
    global game_window_x, game_window_y, game_hwnd
    
    if not game_hwnd:
        # print(f"❌ 錯誤: 遊戲視窗句柄為 None，請先找到視窗。")
        return False

    try:
        rect = win32gui.GetWindowRect(game_hwnd)
        game_window_x, game_window_y, _, _ = rect
        # print(f"✅ 遊戲視窗 '{TARGET_WINDOW_TITLE}' 的左上角座標已更新為: ({game_window_x}, {game_window_y})")
        return True
    except Exception as e:
        print(f"❌ 無法獲取遊戲視窗座標: {e}")
        return False

def on_mouse_click(x, y, button, pressed):
    """滑鼠點擊事件的回調函數。 (此函數用於模擬，實際直接使用 pyautogui.position()) """
    if pressed: # 只處理按下滑鼠鍵的事件
        print(f"\n--- 偵測到點擊 ---")
        print(f"滑鼠絕對螢幕座標: ({x}, {y})")

        # 確保遊戲視窗座標已更新
        if get_game_window_coords():
            relative_x = x - game_window_x
            relative_y = y - game_window_y
            print(f"相對於遊戲視窗左上角的座標: ({relative_x}, {relative_y})")
            print(f"請將這些值作為 GAME_BOARD_OFFSET_X 和 GAME_BOARD_OFFSET_Y 的參考。")
        else:
            print("無法計算相對於遊戲視窗的座標，因為遊戲視窗座標未取得。")
        
        print(f"--- 請繼續測量 (按 Enter 鍵), 或按 Q 鍵退出程式 ---")
        winsound.Beep(800, 200) # 點擊後播放一個提示音

def on_enter_press_callback():
    """Enter 鍵被按下時的回調函數。"""
    global enter_pressed
    if not enter_pressed: # 避免重複觸發
        enter_pressed = True
        winsound.Beep(600, 300) # 播放一個提示音，表示已接收到 Enter
        print("\n--- Enter 鍵已按下！請在 3 秒內點擊遊戲中的目標位置 ---")
        print("倒計時: 3...")
        time.sleep(1)
        print("倒計時: 2...")
        time.sleep(1)
        print("倒計時: 1...")
        time.sleep(1)
        
        # 獲取點擊前的鼠標位置作為參考
        pre_click_x, pre_click_y = pyautogui.position()
        print("現在點擊...")

        # 在 1 秒內等待用戶點擊，如果用戶沒有移動滑鼠，就取 3 秒前的位置
        # 由於無法直接監聽點擊，這裡我們假設用戶點擊後滑鼠會停留在點擊位置
        # 這裡不再等待用戶點擊，而是直接獲取當前滑鼠位置 (假設用戶在倒計時結束時已經點擊了)
        final_x, final_y = pyautogui.position() 
        on_mouse_click(final_x, final_y, None, True) # 模擬點擊事件
        
        enter_pressed = False # 重置標誌，以便下次可以再次觸發
        
def on_q_press_callback():
    """Q 鍵被按下時的回調函數。"""
    print("檢測到 Q 鍵，程式即將退出。")
    winsound.Beep(400, 200) # 退出時播放提示音
    sys.exit(0) # 安全退出程式

# --- 主程式 ---
def main():
    global game_hwnd

    print("--- 棋盤偏移量測量工具 ---")
    print(f"請確保遊戲 '{TARGET_WINDOW_TITLE}' 正在運行並處於活躍狀態。")
    print("將滑鼠移動到遊戲中「棋盤區域最左上角的第一個方塊的左上角」，然後點擊滑鼠左鍵。")
    print("按下 Enter 鍵開始測量，然後立即在遊戲中點擊目標位置。")
    print("按下 Q 鍵可以隨時退出程式。")
    print("\n等待遊戲視窗...")

    # 註冊熱鍵監聽器
    keyboard.add_hotkey('enter', on_enter_press_callback)
    keyboard.add_hotkey('q', on_q_press_callback)

    # 尋找遊戲視窗
    while True:
        game_hwnd = get_window_by_title(TARGET_WINDOW_TITLE)
        if game_hwnd:
            print(f"✅ 找到遊戲視窗: '{TARGET_WINDOW_TITLE}'")
            get_game_window_coords() # 嘗試獲取遊戲視窗初始座標
            break
        else:
            print(f"❌ 無法找到遊戲視窗: '{TARGET_WINDOW_TITLE}'，請確認遊戲是否已開啟且標題正確！1秒後重試...")
            time.sleep(1)

    print("\n--- 現在，請準備在遊戲中點擊目標位置 ---")
    print("當你準備好時，按下 Enter 鍵。")

    try:
        while True:
            # 主循環持續運行，等待熱鍵事件
            time.sleep(0.1) # 減少 CPU 佔用
    except SystemExit: # 捕捉 sys.exit() 異常
        print("程式已正常結束。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
    finally:
        keyboard.unhook_all() # 解除所有熱鍵註冊

if __name__ == "__main__":
    # 檢查 winsound 是否可用 (僅限 Windows)
    if sys.platform != 'win32':
        print("警告: winsound 模組僅在 Windows 系統上可用。聲音提示功能將被禁用。")
        # 替換 winsound.Beep 為一個空函數，避免錯誤
        def dummy_beep(freq, duration):
            pass
        winsound.Beep = dummy_beep

    main()