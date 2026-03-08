# 4DGS Pipeline Tool - 開發進度紀錄

## 專案目標

開發 GUI 工具，完成 Video → Images → 3DGS PLY → UE RAW 的完整轉換流程。
RAW 輸出格式與 Unreal Engine GaussianStreamer Plugin 完全相容。

---

## Phase 1: 核心模組 ✅

### PLY Reader (`app/utils/ply_reader.py`)
- Binary little-endian PLY 讀取器
- 支援動態 property mapping，任何 PLY layout 都能讀
- 輸出 dict: position, sh_dc, sh_rest_r/g/b, opacity, scale, rotation

### Morton Sort (`app/utils/morton.py`)
- COLMAP → UE 座標轉換：`(x,y,z) → (z*100, x*100, -y*100)`
- 3D Morton code 排序，完全複製 UE plugin 邏輯

### PLY → RAW Converter (`app/pipeline/ply_to_raw.py`)
- 精確複製 `GSRawFrameConverter.cpp` + `ThreeDGaussiansLibrary.cpp` 邏輯
- 15 個 RGBA texture（position, rotation, scaleOpacity, sh_0~sh_11）
- 支援 Full (32bit) / Half (16bit) precision
- 產生 metadata.json + sequence.json
- Resume 支援（跳過已轉換的 frame）

### 驗證
- 用 `ref/io-example/` 的範例 PLY 做轉換
- metadata.json 結構與數值一致
- binary 檔案大小完全吻合

---

## Phase 2: Pipeline 整合 ✅

### Video → Images (`app/pipeline/video_to_images.py`)
- ffmpeg 均勻取樣
- 自動計算 step = total_frames // desired_count

### Images → PLY (`app/pipeline/images_to_ply.py`)
- 封裝 ml-sharp `sharp predict` CLI
- 每張圖產生一個 PLY（1,179,648 gaussians/frame）
- GPU ~35s/frame, CPU ~40s/frame

---

## Phase 3: GUI ✅

### Tkinter GUI (`app/main.py`)
- 選影片後自動生成所有路徑
- 三個步驟可單獨執行或一鍵全流程
- 進度條 + ETA + 即時 log
- Threading 避免 GUI 凍結
- Stop 功能（可中斷當前操作）
- 可收合 log 區域
- Estimates 區域顯示預估時間/空間

### RAW 轉換設定
- Sequence Name / FPS / SH Degree
- Precision: Pos/Rot/Scale/SH 各自可選 32bit 或 16bit
- **Contribution-based pruning**: 勾選啟用，設定保留百分比（預設 50%）
- 資料夾名稱自動帶 pruning suffix（如 `_raw_top50`）

---

## Pruning 研究

### Opacity Threshold（已棄用）
- ml-sharp 的 opacity 分布：92% > 0.7，只有 0.1% < 0.05
- Opacity threshold 對 ml-sharp 輸出幾乎無效

### Contribution-based Pruning（目前方案）
- 公式：`contribution = sigmoid(opacity) × volume`
- `volume = exp(scale_0) × exp(scale_1) × exp(scale_2)`
- 排序後保留 top N%
- 測試結果：
  - 100%: 1,179,648 gaussians, 144.2 MB/frame
  - 70%: 825,753 gaussians, 100.9 MB/frame (-30%)
  - **50%: 589,824 gaussians, 72.0 MB/frame (-50%)** ← 目前使用
- 50% 在 UE 中目視效果與 100% 差異不大

---

## 測試資料

### teamlab-1
- 來源：`D:\4dgs test\teamlab-1\teamlab-1.MOV`（3371 frames, ~113 秒）
- Frames: 360 張（均勻取樣，step=9）+ 30 張（影片尾段）= 390 張
- PLY: `teamlab-1_ply/`（390 個, 每個 ~64MB）
- RAW: `teamlab-1_raw_top50/`（390 frames, 50% pruning, 27.42 GB）
- 播放：24 FPS, 16.2 秒

### 效能數據
| 步驟 | 速度 | 備註 |
|------|------|------|
| ffmpeg 取幀 | ~0.1s/frame | 很快 |
| ml-sharp (GPU) | ~35s/frame | 瓶頸 |
| PLY → RAW (50%) | ~1.2s/frame | 含 pruning + morton sort |

### 空間佔用（per frame）
| 格式 | 大小 |
|------|------|
| JPEG frame | ~80 KB |
| PLY | ~64 MB |
| RAW (100%) | ~144 MB |
| RAW (50% pruning) | ~72 MB |

---

## ml-sharp 筆記

- Apple SHARP model，單張圖片 → 3DGS
- 固定輸出 1,179,648 gaussians = (1536/stride)² × num_layers = 768² × 2
- stride=2, num_layers=2 為模型參數，CLI 無法調整
- 會對每張圖做兩次 pass（報告的 total 是實際圖片數 ×2）
- 無 focal length EXIF 時預設 30mm

---

## UE GaussianStreamer 筆記

- 播放 FPS 由 sequence.json 的 `targetFPS` 控制
- Accumulator pattern：每 tick 累加 deltaTime，超過 1/targetFPS 就切下一幀
- 無幀間插值，純離散切換
- N-buffering（2-4 frames）避免載入卡頓

---

## Temporal Smoothing（待實作）

### 問題
ml-sharp 每幀獨立重建，幀間 gaussian 無對應關係，播放時造成：
- 閃爍（flickering）— 同一表面顏色跳變
- 抖動（jittering）— 物體位置微震
- Pop-in/out — gaussian 突然出現/消失

### 方案構想
- 對 Frame N 的每個 gaussian，用 KD-tree 在 Frame N-1、N+1 找最近鄰
- 對 position、SH color、scale、opacity 做加權平均（如 0.2/0.6/0.2）
- 放在 PLY → RAW 之間的中間處理，不改變 PLY 也不改變 RAW 格式

### 挑戰
- 對應關係只能靠空間最近鄰近似，無 ground truth
- 每幀 ~118 萬 gaussians，KD-tree 建構 + 查詢有計算成本
- 快速變化場景（如 teamlab 光影）平滑可能反而糊掉細節
- 移動幅度大時最近鄰可能找錯

### 建議做法
先寫簡單版（只平滑 position + SH，權重可調），拿幾幀測試效果再決定是否全量跑。

---

## 待辦 / 未來方向

### 短期
- [ ] GUI estimates 時間修正（ml-sharp GPU 應為 35s/frame 而非 8s）
- [ ] 測試更多影片素材
- [ ] Temporal smoothing 原型 — 先做簡單版測試效果

### 中期
- [ ] Temporal smoothing 完善（自適應權重、排除大位移 gaussian）
- [ ] RAW temporal compression — delta encoding 減少磁碟空間
- [ ] 更智慧的 pruning（考慮 SH 能量、空間分布等）

### 長期（研究方向）
- [ ] 評估 Deformable 3DGS / Shape of Motion 做真正的 4DGS
  - 優勢：temporal consistency、小模型、可換視角
  - 門檻：需要 per-scene training、需要寫 export PLY 腳本
  - 限制：單視角場景品質有限
  - hustvl/4DGaussians 需要多視角，不適合固定鏡頭場景
  - 這些方法輸出 PyTorch 模型，需寫 export 腳本產出 per-frame PLY
- [ ] 這些方案的 export → PLY → 現有 RAW converter 的銜接
