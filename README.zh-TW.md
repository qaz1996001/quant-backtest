*[English](README.md) ｜ 繁體中文*

# qbt — 回測出錯的地方通常是量測本身

一個可執行的回測框架，用來支撐一句話：**回測結果多半是在陳述你的評估程序，
所以需要被測試的是程序本身。** 未來資料洩漏、切分幾何、交易成本、退化情況下的
評估指標，每一項都有一條在機制被拿掉時會失敗的斷言。

```bash
uv run --with pytest --with lightgbm --with numpy pytest -q      # 110 passed
uv run --with lightgbm --with numpy python examples/run_backtest.py
```

離線執行、合成資料、固定亂數種子。沒有行情源、沒有 API key、沒有快取目錄。

模型用 LightGBM。它是整個 repo 裡最小的一塊，大致就是它該佔的比例。

---

## 這個 repo 在講的那個數字

同一份資料、同一組特徵、同一個模型，兩種評估程序：

```
fit once on everything   Sharpe   6.744   <- the artefact
walk forward, 5 folds    Sharpe   1.592   <- the number
```

6.744 是 notebook 裡 `fit` 呼叫一次、再對同一批 row 呼叫 `predict` 時印出來的
數字。那段程式沒有 bug，是程序有問題。

而誠實的 1.592 還是毛的。加上合理的 7 bps 來回成本變成 **1.208**；零售等級的
25 bps 變成 **0.214**；70 bps 時這個策略**開始賠錢**。這四列直接來自
`examples/run_backtest.py`。

---

## 回測會洩漏的四個地方，以及各自的測試

### 1. 偷看未來的特徵

檢查器不讀原始碼，也不相信 docstring。它先算一次特徵，把切點之後的每一根 bar
換成另一條路徑，重算一次，然後找切點以下有哪個 index 的值動了。

```
shipped features
  ret_1          causal
  ret_5          causal
  vol_20         causal
  momentum_10    causal
  rsi_14         causal
  zscore_60      causal

features written to cheat
  next_bar_return      leaks   first moved index   20   (probe cut 20)
  centred_window       leaks   first moved index   11   (probe cut 20)
  full_sample_zscore   leaks   first moved index    0   (probe cut 20)
  backfilled_warmup    leaks   first moved index    0   (probe cut 20)
```

`qbt/leaky.py` 裡放了四個故意作弊的特徵，測試斷言這四個都要被抓到。一個只跑過
正確程式碼的檢查器，等於沒被測試過。

這四個都不是靠打錯字產生的，而是各有一條很合理的來路：

| 洩漏 | 怎麼混進來的 |
|---|---|
| `next_bar_return` | 標籤的公式被複製去當特徵 |
| `centred_window` | 從訊號處理的 notebook 搬過來的平滑步驟，在那裡置中是正確的 |
| `full_sample_zscore` | `StandardScaler` 在切分之前就 fit 了 |
| `backfilled_warmup` | 為了補圖左邊的空缺而回填 `nan` |

回報的 index 就是診斷資訊。`next_bar_return` 動的是切點後退一格的位置，所以它
剛好往前看一根。兩個標準化的 bug 動的是 index **0**——那是另一種形狀的錯誤：
整個欄位是用序列末端的數字去縮放的。

### 2. 互相碰到的切分

```
fold 0  train [    0,   299]  test [  311,   535]  gap 11
fold 1  train [    0,   524]  test [  536,   760]  gap 11
fold 2  train [    0,   749]  test [  761,   985]  gap 11
```

間隔是 **11 根 bar**：1 根給標籤期間，10 根給 embargo。

「用時間序列交叉驗證」這句話把三種不同的洩漏綁在一起，所以 `split.py` 把它們
拆開：

- 用了測試視窗之後的 bar 來訓練。走查式切分就解決了。
- 用了標籤跨進測試視窗的 bar 來訓練。期間為 `h` 的標籤在 `t` 這根是 `t + h`
  以前所有 bar 的函數，所以訓練集最後 `h` 根已經含有測試期的資訊。這是
  **purge**，長度是精確的。
- 用了緊貼測試視窗的 bar 來訓練。序列相關讓相鄰的 bar 幾乎是同一根。這是
  **embargo**，長度是你選的，不是推導出來的。

只有第一項是把資料按日期排序就解決的。

測試斷言的是可量測的後果：每一折 `gap == horizon + embargo`、embargo 加大則
間隔加大、`embargo=0` 時 purge 仍然生效。一個被記錄下來卻沒被套用的參數，正是
這裡要測的失效模式。

### 3. 名義上可設定、實際上是零的成本

`Costs()` 預設 2 bps 手續費加 5 bps 滑價。預設為零是那種會被不小心留在程式裡的
版本。

關鍵斷言：同一個訊號在零成本與真實成本下，Sharpe 必須**不同**，而量測出來的
成本拖累必須為正。接著還有一條測試說明為什麼這件事重要——一個在 58% 的 bar 上
猜對方向的訊號，**毛的賺、淨的賠**，大約 50 bps 的摩擦對上 13 bps 的優勢。

### 4. 永遠給好看數字的評估指標

| 策略 | 指標必須說什麼 |
|---|---|
| 全部現金、從不交易 | 最大回落 **0**、Sharpe **nan**、Calmar **nan**、勝率 **nan** |
| 反向訊號、每根都做反 | Sharpe **-16.769**、總報酬 **-1.0**、最大回落 **1.0** |
| 買進持有 | Sharpe **0.800**、最大回落 **0.584** |

需要測試的是第一列。平的權益曲線沒有風險調整後報酬；印出 `0.00` 等於在宣稱一個
沒人承擔過的風險。`nan` 會讓讀到它的人注意到。

年化在每個地方都是參數。`252` 只以 `BARS_PER_YEAR["daily"]` 的值出現過，不會
寫死在任何公式裡，而且實際用到的 `bars_per_year` 會帶在報告裡。用小時 bar 算出
來、卻當成日頻在讀的 Sharpe，會差 **2.5 倍**。

---

## 漂移偵測放在這裡的理由

回測說的是一個策略在某段歷史上有效。它完全沒有說今天的輸入還像不像那段歷史。
`drift.py` 對特徵與滾動績效跑滑動視窗 Z-score。

資料產生器在第 **600** 根 bar 刻意製造一次波動度 regime 變化，所以偵測器可以
對著 ground truth 檢查，而不是對著一張圖：

```
ret_1            ok                 z=  -0.05
vol_20           drift              z=   5.81  |z| over 3.0
momentum_10      ok                 z=  -0.10
```

兩種「拒絕回答」被測得和偵測本身一樣仔細：

```
vol_20           insufficient_data  z=    n/a  baseline has 5 points, need 30
stuck_sensor     insufficient_data  z=    n/a  baseline has zero variance
```

十個點的視窗與完全平坦的基線，都讓 z-score 無定義。在這裡回報 `ok`，就是監控
系統對著一張空的基線表亮綠燈亮了一週的機制。完全平坦的基線，也正是感測器卡住
或快取值重複時的樣子。

特徵漂移是**雙邊**的：低於訓練範圍三個標準差的輸入，對模型來說和高三個標準差
一樣陌生。績效漂移是**單邊**的，`higher_is_better` 先把方向正規化一次，所以
不論 Sharpe 還是回落，負的 z 一律讀作「變差」。留兩條比較分支，就是留兩個機會
讓不等式寫反，而一個在回落上寫反的監控，會剛好在最該叫的時候保持安靜。

---

## 測試涵蓋了什麼

**110 條測試。** 模組約 1,500 行，測試約 1,100 行，這就是這個 repo 想主張的
比例。

| 檔案 | 測試數 | 釘住了什麼 |
|---|---|---|
| `test_lookahead.py` | 17 | 每個出貨的特徵都是因果的；四個洩漏全被抓到；洩漏距離對得上 bug；暖機 row 是被丟掉而不是被填值 |
| `test_metrics.py` | 17 | 全現金給 0 回落與 nan Sharpe；反向訊號給鏡像的 Sharpe；年化是參數；常數報酬序列沒有 Sharpe |
| `test_drift.py` | 16 | 過短與全平的基線回報 `insufficient_data`；第 600 根的 regime 變化被找到；變好不算漂移 |
| `test_split.py` | 16 | 切分不共用任何 row；測試折連續且覆蓋到尾端；間隔等於 purge 加 embargo；不可能的設定會丟錯 |
| `test_pipeline.py` | 16 | 沒有任何 bar 是被訓練過它的模型預測的；植入的 AR(1) 被找到；**沒有植入訊號時也找不出訊號**；一個洩漏特徵把 Sharpe 推到 13.0 |
| `test_backtest.py` | 13 | 部位在訊號後一根才持有；反手付兩次成本；弱優勢被 50 bps 吃乾 |
| `test_model.py` | 9 | `nan` 輸入被擋下並指名欄位；同種子給完全相同的預測、不同種子不會 |
| `test_reproducibility.py` | 6 | 兩次相同的 pipeline 跑出 byte 等同的淨報酬與完全相同的報告 |

其中兩條是對照組，不是在測被測物：

**負對照。** 在 `autocorr=0.0` 時產生的報酬彼此獨立，什麼都不可預測，走查式的
Sharpe 必須回來很小。pipeline 裡任何一處洩漏——特徵偷看、折次重疊、標籤差一格
——都會在純噪音上產生一個自信的數字，這讓它成為整份測試裡最便宜的回歸測試。

**正對照。** 把一個誠實的特徵換成 `next_bar_return`，再跑一次完全相同的程序：
折次正確、purge 正確、embargo 正確、成本照收。Sharpe 從 **1.592 變成
13.008**，最大回落從 **0.201 變成 0.002**。切分紀律擋不住一個會偷看的特徵，
這就是因果檢查器要當成獨立關卡、而不是寫成註解的原因。

---

## 怎麼跑

```bash
# 測試
uv run --with pytest --with lightgbm --with numpy pytest -q

# 整個範例，或只跑其中幾段
uv run --with lightgbm --with numpy python examples/run_backtest.py
uv run --with lightgbm --with numpy python examples/run_backtest.py leak costs
```

可用的段落：`audit`、`folds`、`leak`、`costs`、`null`、`drift`。輸出是純
ASCII，因為寫這個 repo 的終端機是 CP950，報告裡一個框線字元就是一次崩掉的
執行。

**後端。** LightGBM 是預期的後端，透過它原生的 `Dataset` / `train` API 驅動。
第一版用的是 `LGBMRegressor`，而它在缺少 scikit-learn 時會丟
`LightGBMError`——那會讓一個原本只需要 numpy 與 lightgbm 的 repo 多一個安裝
依賴。原生 API 沒有這個限制。

如果環境沒有可用的 lightgbm wheel，封裝會退回
`sklearn.ensemble.GradientBoostingRegressor` 並且講出來——`which_backend()`
會印在每次範例執行的開頭，測試也會斷言它。**兩個後端下 110 條測試都通過。**
靜默退回的後果是兩台機器跑出不同的數字，而畫面上沒有任何東西解釋原因。

---

## 寫這個 repo 時真的出錯的地方

留著，因為這些才是真正在這裡學到的部分。

**一個探測切點不夠。** 因果檢查器原本只在序列中點切一次，四個作弊特徵抓到
三個。`backfilled_warmup` 乾淨通過，因為到第 750 根已經沒有暖機值可以回填——
它只污染前六十根。`default_cuts()` 現在會探測六個切點，包含第 20 根這個早期
切點。

**`sd == 0.0` 是錯的退化保護。** 一條斷言「常數報酬序列沒有 Sharpe」的測試
失敗了。500 個相同的 1 bp 報酬，樣本標準差約為 **1e-20**，因為算出來的平均值
不會剛好等於輸入值。它不是零，所以保護放它過去，比值年化後跑出大約 **1e16**。
修法是改用相對下限 `ZERO_DISPERSION = 1e-12`。

**完美先知證明不了成本重要。** 成本測試的第一版用了一個精準知道下一根報酬的
訊號，斷言它在 50 bps 下會不賺錢。它失敗了：這條序列上平均絕對單根報酬約
**80 bps**，所以先知在任何合理摩擦下都還是大賺。現在這條測試用 58% 的命中率
——成本真正會咬到的區域——而前提被修正這件事本身就是收穫。

**擾動必須讓價格保持為正。** 用 `close * (1 + 0.5 * randn)` 擾動會把一些 bar
壓成負數，所有對數報酬特徵回傳 `nan`，檢查器於是報出一堆其實是「餵了不可能的
序列」造成的假洩漏。現在是 `close * exp(0.5 * randn)`。

**把植入訊號調強是一個判斷。** AR(1) 係數一開始是 0.06，比較接近真實世界，
走查式 Sharpe 跑出 **-0.605**——誠實，但當示範很沒用，因為一個賠錢的策略讓
成本階梯和漂移那一段都很難讀。現在是 0.25，**比任何流動市場都強得多**，
`synth.py` 的 docstring 有寫明。`autocorr=0.0` 的負對照存在，就是為了不讓這個
選擇變成一個聲稱。

---

## 限制

- **合成資料不是市場。** 產生器是帶 AR(1) 項的 regime 切換隨機漫步。它沒有
  像樣的厚尾、除了 regime 排程以外沒有波動度叢聚、沒有跳空、沒有隔夜缺口、
  沒有財報日。在這裡有效的策略，不代表在任何其他地方有效。
- **植入的 AR(1) 0.25 強得不現實。** 日頻股票報酬的自相關頂多百分之幾。這個
  數字被調高是為了讓框架有東西可以找，它不是任何東西的估計值。
- **單一資產、單一部位、沒有投組。** 沒有橫斷面排序、持倉之間沒有相關性、
  沒有波動度目標的部位大小、沒有槓桿限制、沒有保證金。
- **成本是換手率的線性函數。** 真實滑價取決於委託量對上掛單簿深度、時段、
  以及推動價格的人是不是你。市場衝擊是非線性的，這裡沒有建模。
- **完全沒有微結構。** 沒有把買賣價差當成獨立量、沒有部分成交、沒有排隊位置、
  沒有交易所費用表、空方沒有借券成本與可借性。部位假設在下一根 bar 的收盤
  成交。
- **沒有多重檢定校正。** 走查式程序給的是*一組*策略設定的誠實估計。試四十組
  再報最好的那組，需要 deflated Sharpe 或 reality check，兩者都不在這裡。
- **標籤是單根前瞻報酬。** Triple-barrier 標籤、meta-labelling、依唯一性加權
  取樣都屬於這個問題，而都沒有實作。
- **漂移偵測假設基線近似常態。** 對厚尾特徵跑 z-score 會過度觸發。改用基於
  排序或分位數的統計量會是正確的升級方向。

---

## 這些想法從哪裡來

時間序列特徵工程、滑動視窗 Z-score 漂移偵測、評測閘門，以及「測量測本身而不是
只測模型」這個紀律，都來自我在工業感測資料與一個 agent 平台上做的生產環境 ML
工作。那些程式碼是私有的。

**這個 repo 沒有使用任何前雇主的程式碼。** 它是從零寫的，跑在合成資料上，
領域也是我沒有專有工作成果的領域。延續過來的是推理過程，寫在
[docs/design.md](docs/design.md)。

另外兩個 repo 把同樣的做法用在不同問題上：`idemingest`（冪等重放與失敗分類）
與 `agent-control-plane`（agent pipeline 的決定性控制平面，漂移模組的形狀就是
從那裡來的）。

MIT 授權。
