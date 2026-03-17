# BrownDust II 連連看自動化機器人 (BD2 Little Game Bot)

這是一個專為《棕色塵埃 2》(BrownDust II) 小遊戲「連連看」設計的自動化 Python 腳本。它結合了 **模板匹配** 與 **動態特徵分類** 技術，能夠精準識別遊戲畫面中的方塊，並自動計算路徑進行消除。

## ✨ 主要功能

*   **雙重掃描模式**：
    *   **模板匹配模式 (Template Mode)**：使用預先建立的圖庫進行精準識別，適合已知樣式。
    *   **動態分類模式 (Dynamic Classification)**：即時分析方塊的「感知雜湊 (pHash)」與「顏色特徵」，無需圖庫即可識別新出現的方塊變體（特別是同形狀但不同顏色的方塊）。
*   **智能卡死檢測**：當機器人發現連續多次嘗試消除相同位置失敗（卡死 > 5 次）時，會自動切換掃描模式以嘗試突破僵局。
*   **路徑演算法**：內建 BFS 搜尋演算法，確保只點擊符合規則（轉彎 2 次以內）的路徑。
*   **自動化操作**：包含視窗偵測、截圖、路徑計算與滑鼠點擊。

## 🛠️ 安裝需求

請確保您的環境已安裝 Python 3.x，並執行以下指令安裝必要套件：

```bash
pip install opencv-python numpy pyautogui pywin32 keyboard imagehash Pillow
```

## 🚀 使用說明

1.  **前置設定**：
    *   打開遊戲並進入連連看畫面。
    *   執行 `measure_board_offset.py`（若有）來測量棋盤左上角座標，或手動修改主程式中的 `GAME_BOARD_OFFSET_X` 和 `GAME_BOARD_OFFSET_Y`。
2.  **啟動機器人**：
    *   以管理員身分執行 CMD 或 PowerShell。
    *   執行指令：`python "lian_lian_kan_bot(堪用).py"`
3.  **操作熱鍵**：
    *   `F3`: **啟動** 自動消除
    *   `F4`: **暫停** 機器人
    *   `F5`: 開啟/關閉除錯圖片保存
    *   `Q`: 退出程式

## 🔄 程式運作流程圖

```mermaid
graph TD
    Start[啟動程式] --> Init[初始化: 載入設定與模板庫]
    Init --> WaitForStart{等待 F3 啟動}
    WaitForStart -- F3 Pressed --> LoopStart[開始主迴圈]
    
    LoopStart --> Capture[截取遊戲視窗畫面]
    Capture --> CheckMode{檢查當前模式}
    
    CheckMode -- Dynamic --> DynamicScan[動態分類掃描<br/>(pHash + 顏色特徵)]
    CheckMode -- Template --> TemplateScan[模板匹配掃描<br/>(預存圖庫)]
    
    DynamicScan --> Identify[識別棋盤狀態]
    TemplateScan --> Identify
    
    Identify --> FindPairs[計算可消除對子<br/>(BFS 搜尋 <= 2 轉彎)]
    
    FindPairs --> CheckStuck{檢測卡死狀態<br/>(重複對子 > 5?)}
    
    CheckStuck -- Yes --> SwitchMode[切換掃描模式<br/>Dynamic <--> Template]
    SwitchMode --> LoopStart
    
    CheckStuck -- No --> ExecClick{有可消除對子?}
    
    ExecClick -- Yes --> ClickPairs[模擬滑鼠點擊消除]
    ClickPairs --> LoopStart
    
    ExecClick -- No --> LoopStart
```

## 📁 專案結構

*   `lian_lian_kan_bot(堪用).py`: 主程式邏輯。
*   `accumulated_templates_library/`: 存放已知方塊的圖片模板。
*   `measure_board_offset.py`: 輔助工具，用於校正座標。
