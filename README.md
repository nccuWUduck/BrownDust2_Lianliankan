

# BrownDust II 連連看自動消除外掛 (Lianliankan Auto-Solver Bot)

這個專案是一個用 Python 編寫的連連看遊戲自動消除外掛，專為遊戲《棕色塵埃2》(BrownDust II) 中的連連看小遊戲設計。它能夠自動識別棋盤上的方塊，找出可消除的對子，並模擬滑鼠點擊進行消除。

**重要提示：此程式設計用於 1920x1080 解析度的 Windows 系統，且「縮放與版面配置」設定為 125%。若你的系統設定不同，可能需要調整棋盤參數以確保精確度。**

## 輔助工具說明與已知問題

本專案包含以下主要檔案和輔助工具：

1.  `crop_templates.py`: 用於**製作方塊模板**，並將其保存到 `accumulated_templates_library` 資料夾中。這是推薦的模板製作方式。
2.  `XY_pic_template.py`: 舊版的手動模板製作工具。**不推薦使用**，因為其製作的模板大小可能不一致，影響識別準確性。
3.  `measure_board_offset.py`: 輔助測量遊戲棋盤在遊戲視窗中的偏移座標 (X, Y)。運行此工具後，將滑鼠移動到棋盤的左上角，按下 Enter，你會聽到兩次蜂鳴聲，表示座標已獲取。這些座標應填入 `lian_lian_kan_bot.py` 的配置中。
4.  `lian_lian_kan_bot.py` (主程式檔案): 這是核心程式，負責遊戲視窗擷取、方塊識別、路徑搜尋、模擬點擊以及除錯顯示等功能。
    * **已知問題**: 程式碼尚未完全最佳化，有時可能會點擊到無法消除的配對。
    * **空方塊處理**: 舊的 `template_empty.png` 模板已放棄，目前是透過直接判斷純色來識別空方塊。程式中提到的 `template_empty_blue_green.png` 和 `template_empty_red_striped.png` 是已廢棄的模板方案，目前不使用也不會進行修改。
    * **挑戰模式背景**: 針對挑戰模式 (Challenge) 中出現的紅色背景，是利用 `persisted_e_board` 機制來「硬幹」處理，確保程式可以將這些區域視為可通過的空位。
    * **原理**: 切割圖片，縮兩圈辨識是否純色，縮一圈切成template，主程式中self.NORMAL_TEMPLATE_WIDTH需要手動改，與self.INNER_TILE_PADDING已經切割開來了

**`persisted_e_board` 原理：** 當一個（非空、非鎖定、非錯誤的）正常方塊，在下一輪掃描中變為（空、鎖定、錯誤）的方塊時，`persisted_e_board` 會將該位置標記為「可通過的空位 `E`」，從而幫助路徑搜尋算法正確判斷棋盤狀態。

## 專案特色

* **圖像識別**: 運用 OpenCV 和感知哈希 (Perceptual Hashing) 技術來識別遊戲棋盤上的方塊類型。支援多種方塊、鎖定方塊以及兩種不同的空方塊背景（透過純色判斷）。
* **路徑搜尋**: 使用廣度優先搜尋 (BFS) 演算法，在 0、1 或 2 個轉彎的限制下，尋找兩點之間的可消除路徑。
* **智慧棋盤狀態管理**: 透過追蹤方塊的變化和 `persisted_e_board` 機制，智慧判斷哪些位置已經被消除（即使遊戲畫面未立即刷新），避免誤點或重複判斷。
* **可擴展模板系統**: 透過 `accumulated_templates_library` 資料夾管理所有識別模板，方便增減或更新遊戲方塊。
* **除錯模式**: 提供選項保存掃描過程中的圖片和識別結果，方便除錯和分析。圖片會保存在 `debug_board_scans` 資料夾中。
* **熱鍵控制**: 方便地啟動 (F3)、暫停 (F4) 和退出 (Q) 外掛，以及開關除錯圖片保存 (F5)。

## 運行環境

* **作業系統**: 僅支援 Windows (因為使用了 `win32gui` 模組來獲取遊戲視窗)。
* **Python 版本**: Python 3.8+ (推薦 3.9 或更高版本)
* **遊戲**: 《棕色塵埃2》 (BrownDust II) - 連連看小遊戲模式

## 安裝與設定

### 1. 克隆專案

首先，將本專案從 GitHub 克隆到你的本地電腦。打開你的終端機 (例如 Git Bash 或 PowerShell)：

```bash
git clone [https://github.com/nccuWUduck/BrownDust2_Lianliankan.git](https://github.com/nccuWUduck/BrownDust2_Lianliankan.git)
cd BrownDust2_Lianliankan
````

### 2\. 安裝依賴套件

進入專案資料夾後，使用 pip 安裝所有必要的 Python 套件：

```bash
pip install opencv-python numpy pyautogui pillow imagehash keyboard pywin32
```

### 3\. 配置遊戲視窗標題和棋盤參數

打開 `lian_lian_kan_bot.py` 檔案，找到頂部的 `--- 配置參數 ---` 區塊，根據你的實際遊戲視窗和遊戲內連連看棋盤位置，精確修改以下變數：

```python
TARGET_WINDOW_TITLE = "BrownDust II" # 確保與你的遊戲視窗標題完全一致

# 遊戲棋盤在遊戲視窗中的相對位置和大小
# 這些值需要根據你的遊戲視窗截圖進行精確測量。
# 建議使用 Windows 內建的「剪取工具」(Snipping Tool) 或其他截圖工具，
# 截取**整個遊戲視窗**，然後測量連連看棋盤區域相對於遊戲視窗左上角的像素偏移量。
GAME_BOARD_OFFSET_X = 445 # 遊戲棋盤左上角在遊戲視窗內的 X 座標 (預設值，請依實際測量調整)
GAME_BOARD_OFFSET_Y = 259 # 遊戲棋盤左上角在遊戲視窗內的 Y 座標 (預設值，請依實際測量調整)

# 方塊的近似大小
# 如果遊戲內方塊大小有變，需要調整 (目前已根據常見情況設定)
TILE_WIDTH = 64
TILE_HEIGHT = 78

# 棋盤的行列數 (目前已確認為 7 行 16 列，若遊戲更新有變動請修改)
BOARD_ROWS = 7
BOARD_COLS = 16
```

**重要提示：`GAME_BOARD_OFFSET_X` 和 `GAME_BOARD_OFFSET_Y` 是此外掛能否正常運作的關鍵。請務必仔細測量！你可以使用 `measure_board_offset.py` 這個輔助工具來協助獲取這些座標。請確保你的 Windows 縮放設定為 125%，以符合預設參數。**

### 4\. 準備方塊模板

外掛依賴於 `accumulated_templates_library` 資料夾中的方塊圖片模板進行識別。

  * 請確保此資料夾存在於你的專案根目錄中。
  * 它應該包含你遊戲中所有不同類型的連連看方塊圖片，命名格式為 `template_XXX.png` (例如 `template_001.png`, `template_002.png` 等)。
  * 資料夾中還應包含用於識別鎖定方塊的特定模板：`template_locked_1.png` 到 `template_locked_4.png` (鎖定方塊)。

**注意：空方塊目前是透過純色判斷，不依賴 `template_empty.png`、`template_empty_blue_green.png` 或 `template_empty_red_striped.png` 這些模板。**

如果遊戲更新導致方塊圖案變化，你可能需要使用 `crop_templates.py` 或手動截圖更新這些模板圖片。

## 運行外掛

1.  確保《棕色塵埃2》遊戲已經開啟，並進入連連看小遊戲介面。

2.  打開你的終端機，進入專案資料夾後，運行主程式：

    ```bash
    python lian_lian_kan_bot.py
    ```

3.  程式啟動後，會在命令列顯示提示訊息，並嘗試找到遊戲視窗。

4.  程式運行時，會彈出一個實時顯示識別結果的 Debug 視窗，它會覆蓋在遊戲上方。

5.  **操作熱鍵**:

      * `F3`: 啟動外掛，開始自動消除方塊。
      * `F4`: 暫停外掛的自動操作。
      * `F5`: 開啟/關閉除錯圖片保存功能。開啟後，程式會在 `debug_board_scans` 資料夾中保存每輪掃描的詳細截圖和識別結果。
      * `Q`: 退出程式。

## 除錯與問題排查

如果外掛未能正常運作或出現錯誤，請參考以下常見問題排查步驟：

  * **程式當機或錯誤訊息**:
      * 檢查控制台輸出的錯誤訊息 (Traceback)。錯誤訊息通常會指出問題所在的檔案和行號。
      * 常見錯誤如 `NameError` (變數未定義) 或 `IndexError` (索引超出範圍)，這通常是由於棋盤參數 (`GAME_BOARD_OFFSET_X`, `GAME_BOARD_OFFSET_Y`, `TILE_WIDTH`, `TILE_HEIGHT`, `BOARD_ROWS`, `BOARD_COLS`) 設定不正確，導致截圖或圖片處理時發生問題。請仔細檢查並重新測量。
      * 如果程式卡住或反應緩慢，可以嘗試將 `SCAN_INTERVAL` (在 `lian_lian_kan_bot.py` 中) 增加到 `0.5` 秒或 `1.0` 秒，以降低 CPU 負荷。
  * **無法找到遊戲視窗**:
      * 請確認你的遊戲視窗標題與 `lian_lian_kan_bot.py` 中 `TARGET_WINDOW_TITLE` 的值完全一致。大小寫和空格都很重要。
      * 確保遊戲視窗不是最小化狀態。
  * **方塊識別錯誤**:
      * **開啟 `F5` 除錯圖片保存功能。** 程式會在 `debug_board_scans` 資料夾中保存每次掃描的詳細截圖 (`scan_XXXXX_board_with_labels.png`) 以及每個單獨裁剪的方塊圖片 (`scan_XXXXX_rYY_cZZ.png`)。
      * 檢查 `debug_board_scans/unknown_tiles` 資料夾中是否有未能識別的方塊。這表示你需要為這些方塊添加新的模板到 `accumulated_templates_library`。
      * 檢查 `debug_board_scans/tiles` 資料夾中的圖片，確認方塊是否被正確地從遊戲畫面中裁剪出來。
      * 如果空方塊被誤判為實體方塊（或反之），可能需要調整 `EMPTY_TILE_COLOR_TOLERANCE`。
      * 如果實體方塊被誤判，可能需要調整 `HASH_THRESHOLD`，或者更新相應的 `template_XXX.png` 模板。
  * **無法消除方塊或點擊到無法消除的對子**:
      * 首先確認方塊識別是否正確。如果識別錯誤，外掛自然無法找到匹配對。
      * 檢查 `debug_board_scans` 中顯示的識別結果。確認相同圖案的方塊是否都被正確識別為同一個 ID。
      * 確認遊戲中確實存在可消除的路徑（0彎、1彎或2彎）。BFS 算法是嚴格按照規則搜尋的。
      * 盡管使用了 `persisted_e_board` 機制來處理遊戲刷新延遲，如果仍有此類問題，可能需要進一步檢查 `is_passable` 函數的邏輯或 `persisted_e_board` 的更新時機。

## 未來可能的改進方向

  * **更智能的模板管理**: 實現一個更方便的模板生成/更新工具，甚至可以自動學習新方塊。
  * **動態適應性**: 自動檢測遊戲視窗位置和棋盤大小，無需手動配置，增加通用性。
  * **性能優化**: 探索更高效的圖像處理和搜尋算法，減少資源佔用和提高反應速度。
  * **多語言支援**: 讓程式輸出支援更多語言。

## 貢獻

歡迎任何形式的貢獻，包括但不限於：

  * 錯誤報告和修復
  * 功能改進或新功能開發
  * 代碼優化
  * 更新遊戲版本後的兼容性維護
  * 更新或新增方塊模板

如果你想貢獻，請先創建一個 [Issue](https://www.google.com/search?q=https://github.com/nccuWUduck/BrownDust2_Lianliankan/issues) 描述你的想法或問題，然後提交一個 [Pull Request](https://www.google.com/search?q=https://github.com/nccuWUduck/BrownDust2_Lianliankan/pulls)。

## 許可證

這個專案遵循 [你的許可證名稱] 許可證。詳情請見 [LICENSE](https://www.google.com/search?q=https://github.com/nccuWUduck/BrownDust2_Lianliankan/blob/main/LICENSE) 檔案。
*(請替換為你選擇的開源許可證，例如 `MIT License`。如果你沒有 `LICENSE` 檔案，請在 GitHub 上創建一個，或者刪除最後一句)*

## 鳴謝

  * [OpenCV](https://opencv.org/) - 強大的電腦視覺庫
  * [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) - 程式化控制滑鼠和鍵盤
  * [Pillow](https://www.google.com/search?q=https://python-pillow.org/) - 圖像處理庫
  * [ImageHash](https://github.com/JohannesBuchner/imagehash) - 圖像感知哈希庫
  * [Keyboard](https://github.com/boppreh/keyboard) - 監聽和發送鍵盤事件
  * [PyWin32](https://pypi.org/project/pywin32/) - Windows API 擴展
  * [Gemini](https://gemini.google.com/) - 協助生成和優化此 README 內容

<!-- end list -->

```
```