# BrownDust II 連連看自動化機器人 (BD2 Little Game Bot)

這是一個專為《棕色塵埃 2》(BrownDust II) 中的連連看小遊戲設計的自動化 Python 腳本。它利用圖像識別技術來辨識棋盤上的方塊，並自動計算路徑進行點擊消除。

## 專案概述
- **主要語言**: Python
- **核心技術**: 
    - `OpenCV (cv2)`: 圖像處理與模板匹配。
    - `PyAutoGUI`: 模擬滑鼠點擊與螢幕座標獲取。
    - `PyWin32 (win32gui, win32ui)`: Windows 視窗控制與後台截圖。
    - `keyboard`: 全域熱鍵監控（例如：停止腳本）。
    - `imagehash`: 感知雜湊，用於比對相似的方塊圖片。

## 目錄結構
- `lian_lian_kan_bot(堪用).py`: 核心自動化腳本，包含棋盤掃描、路徑演算法與自動點擊。
- `measure_board_offset.py`: 座標測量工具，協助使用者定位遊戲棋盤在視窗中的絕對/相對位置。
- `crop_templates.py`: 方塊模板裁切工具，用於從截圖中自動分割出各個方塊並儲存。
- `accumulated_templates_library/`: 存放各種方塊的圖片模板庫。
- `temp_raw_tiles/`: 執行過程中產生的臨時方塊圖像。

## 快速開始

### 1. 環境準備
確保已安裝必要的 Python 套件：
```bash
pip install opencv-python pyautogui pywin32 keyboard imagehash Pillow
```

### 2. 座標設定
在執行機器人前，必須確保棋盤座標正確：
- 執行 `python measure_board_offset.py`。
- 依照提示在遊戲視窗中點擊棋盤左上角。
- 將輸出的 `GAME_BOARD_OFFSET_X` 與 `GAME_BOARD_OFFSET_Y` 填入 `lian_lian_kan_bot(堪用).py`。

### 3. 執行機器人
確保遊戲視窗標題為 `BrownDust II` 且處於前台（或可見狀態）：
```bash
python "lian_lian_kan_bot(堪用).py"
```

## 開發規範與慣例
- **變數命名一致性**: 
    - 修改或新增功能時，必須嚴格檢查類別屬性（如 `self.BOARD_TILE_GAP_X`）與全域配置（如 `TILE_GAP_X`）的對應關係。
    - 優先使用類別內部定義的屬性（帶有 `self.` 或 `bot_instance.` 前綴的變數）以確保封裝完整性。
    - 在進行大規模替換或重構後，必須全文搜尋舊變數名稱，確保無遺留的舊調用。
- **平行化處理**: 
    - 圖像辨識邏輯應優先考慮使用 `concurrent.futures` 進行平行化，以維持高 FPS。
- **路徑搜尋**: 
    - 連連看路徑演算法應統一使用基於轉彎次數的 BFS 搜尋，並按轉彎次數（0 -> 1 -> 2）排序優先級。

## 注意事項
- 請在法律與遊戲規範允許的範圍內使用此自動化工具。
- 本腳本針對特定解析度與縮放比例設計，若視窗大小變動可能需要重新測量。
