# Astro Party 實作與重現

## 目前交付結果

- checkpoint：`models/model_astro_party.zip`，run `6o6gk5el` 的 40,960 步版本。
- 原版公開評分：Rusher 95%、Sniper 91%，10/10。
- 新保留種子 50000–50099：Rusher 92%、Sniper 87%。
- 驗證種子 10000–10099：90%、84%；因此不宣稱所有種子集均達標。
- 主要訓練曲線已同步並由 API 驗證：[W&B 訓練 run](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/6o6gk5el)。
- 中文報告在 `../output/pdf/Astro_Party_Report.pdf`，可編輯版本為 `REPORT_TW.md`。
- 自動核准審查拒絕批量同步舊 run，以及額外上傳私有模型 / 報告 / 逐局資料。主要訓練 run 已上傳；額外 artifacts 需使用者明確批准後才執行 `experiments/publish_results.py`。
- W&B 原 run 的模型 artifact 是驗證 best（16,384 步），**本次應使用本機 `models/model_astro_party.zip`（40,960 步）**；最終模型 artifact 尚待額外上傳批准。

## 閱讀順序

1. 根目錄 `My_Plan.md`：目標、假設與實驗順序。
2. `envs/train_astro_party_env.py`：雙對手抽樣與 reward。
3. `train_astro_party.py`：`DefenseBot` → `TacticalFeatures` → `behavior_clone` → `train`。
4. `results/` 與 `wandb/`：實際對戰紀錄與訓練紀錄。
5. 最終報告：包含失敗實驗、實測結果與方法限制。

## 方法怎麼理解

這是**規則輔助的 PPO**，不是完全從零學會戰鬥。

`DefenseBot` 根據原始 84 維觀測計算可解釋的控制建議：提前瞄準、遮擋時繞行星、追擊逃生員、逃生躲藏，以及模擬候選轉向 / 推進下的子彈距離。它只使用觀測與公開物理常數，沒有讀取測試種子、對手標籤或隱藏狀態。

`TacticalFeatures` 把 3 維控制建議接上前 26 維狀態，形成 29 維特徵。SB3 內建 `MlpPolicy` 的 actor 為線性層，critic 是 64–64 的 Tanh 網路；共 6,238 個可訓練參數。特徵計算的模擬成本不反映在參數數量內，因此「小模型」不代表完全沒有規則計算成本。

先以教師資料做行為模仿初始化，再執行真正的 PPO rollout、GAE、clipped policy loss、value loss 與梯度更新。可選 `AnchoredPPO` 在每次 PPO 更新後把 actor 權重向初始化收縮，以避免微小瞄準偏移破壞表現。這只改學習過程；最終仍由原版 `PPO.load()` 與 `predict()` 執行。

不應把此方法描述成「PPO 自己發現所有戰術」，也不應把教師成績當成最終模型成績。若課堂投影片另禁止規則特徵，需改走 `--features geometry` 或 `raw` 並重新訓練，現有短跑不保證達標。

## CPU 環境

作業根目錄的 `.venv311` 是本次實際使用的環境。完整套件版本見 `requirements-lock.txt`。以 Python 3.11、CPU PyTorch 2.9.1、SB3 2.9.0 執行。

從根目錄進入遊戲：

```powershell
cd astro_party
```

目前 Python 本身的預設值已經改為最終通過版本的設定，所以最簡單的重現指令就是：

```powershell
python train_astro_party.py
```

這會從頭收集教師資料、做模仿初始化與 40,960 步 PPO，輸出 `models/model_astro_party.zip`。不會載入舊 checkpoint；只有明確指定 `--resume` 才會續訓。W&B 預設為 `auto`：已有登入才用 online，沒有憑證就離線記錄，不會要求助教輸入 API key。模型上傳 W&B 是額外的 `--upload-model` 選項，預設不傳模型檔。

已在 `fast_cover_reproduce/` 的乾淨提交式目錄，使用定稿的同一份訓練檔實際執行裸命令，重訓模型全部 policy tensors 與 reference 相等（最大絕對差 0）。ZIP 的 metadata / 內嵌程式不同，故不要求 ZIP 本身的 SHA-256 相同。紀錄見 `results/reproduction_check.json`，其中也保存訓練主檔 / 環境檔雜湊。此驗證使用本機 Python 3.11 與鎖定套件環境。

提交格式中訓練環境檔與訓練主檔平放，現已同時支援平放的 `train_astro_party_env.py` 與 starter repo 的 `envs/train_astro_party_env.py`。遊戲核心與 baseline 仍由原版環境提供，不依賴 `experiments/` 或任何舊模型。

## 掩體射擊實驗

```powershell
python train_astro_party.py --controller cover
```

此實驗另外輸出 `models/model_astro_party_cover.zip`，不覆寫原本通過 baseline 的模型。它使用中央行星作掩體，補滿彈匣後側移探頭，射擊後退回；暴露時加速退避，近距離壓力下保留原反擊，最後仍執行原本的 60 步避彈搜尋。原始觀測沒有 elapsed time，開局 3–5 秒僅以移動進度 / 速度近似，不是精確計時器。

觀看實驗模型：

```powershell
python watch_astro_party.py --model models/model_astro_party_cover.zip --opponent sniper --seed 0
```

掩體戰術是否更強要以獨立對戰結果判斷；試驗結果完成前不取代預設的已通過版本。

本次實測已完成：種子 61000–61099，各對手各 100 局，原防禦版 Rusher / Sniper 為 92% / 88%，掩體版為 86% / 74%。掩體版對 Sniper 的開局 5 秒出艙率由 14% 降到 5%，但平手率由 1% 升到 11%，因此保留作為試驗而不取代原預設。這版以「補滿彈匣」作為探頭訊號，確保射後有一段退避時間；不是精確計時器。

實際模型的 200 次避彈檢查均改善預測距離，中位改善約 31.5 px；見 `results/cover_trained_dodge_check.json`。成對對戰紀錄見 `results/cover_model_comparison.json`。

- [一鍵重現 W&B](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/ghme0u0w)
- [掩體版訓練 W&B](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/cj240jwb)
- [成對評估 W&B](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/em6va3hq)

重現 60 步閃避版本：

```powershell
./run_training.ps1 -Mode online -Epochs 5
```

自己開一個訓練、另外存檔：

```powershell
./run_training.ps1 -Mode online -Epochs 5 -SavePath models/my_training_checkpoint
```

此命令使用同一組參數：先收集 10,000 個教師步並做 120 輪模仿初始化，再做 5 × 8,192 = 40,960 個 PPO 環境步。訓練完成後輸出 `models/my_training_checkpoint.zip`；終端機會顯示該次 W&B run 網址。

或直接執行（先指定 W&B 可寫目錄）：

```powershell
$env:WANDB_DATA_DIR = "$PWD/.wandb-data"
$env:WANDB_CACHE_DIR = "$PWD/.wandb-cache"
../.venv311/Scripts/python.exe train_astro_party.py --name defense-ppo-h60 --features tactical --controller defense --near 500 --far 700 --horizon 60 --bc-steps 10000 --bc-epochs 120 --epochs 5 --steps-per-epoch 8192 --eval-episodes 50 --learning-rate 0.000001 --select-last
```

訓練資料以 100042 為 RNG 初始種子；每局各有 50% 機率打 Rusher / Sniper。訓練驗證從 10000 起。初期模型未通過後，公開 0–99 被用於開發與選模；最終選擇 40,960 步的最後 checkpoint，因此重現命令加上 `--select-last`。另保留 50000–50099，在固定最終模型後才測試。

## W&B

預設 project 為 `astro-party-hw3`，登入後使用帳戶的預設 entity，可加 `--entity` 指定。

```powershell
../.venv311/Scripts/wandb.exe login
./sync_wandb.ps1
```

API key 僅輸入本機登入提示，不寫進程式、報告或聊天。`--wandb-mode offline` 可在無憑證時保留紀錄；**offline 目錄存在不代表已上傳雲端**。同步後以終端輸出的 run 網址開啟圖表。

離線時也可以立即查看同一批訓練的 TensorBoard 圖表（這不代表 W&B 已同步）：

```powershell
../.venv311/Scripts/tensorboard.exe --logdir runs
```

再於瀏覽器開啟 `http://localhost:6006`。

`sync_wandb.ps1` 預設只處理已選定的主要訓練 run；同步其他 run 需明確指定 `-RunPath`，不會自動批量上傳整個資料夾。

建議在 W&B 查看：

- `eval/rusher_win_rate`、`eval/sniper_win_rate`：橫軸選 `global_step`。
- `eval/*_draw_rate`：檢查策略是否只是拖到平手。
- `rollout/ep_rew_mean`、`rollout/ep_len_mean`：訓練 reward 與局長。
- `train/policy_gradient_loss`、`train/value_loss`、`train/approx_kl`、`train/std`。
- `bc/mse` 對 `bc/epoch`：僅代表教師動作擬合，不代表勝率。
- 模型 artifact 與逐局 evaluation table。

前兩次 pilot 的訓練紀錄完整，但結尾 artifact 暫存曾因 Windows 目錄權限失敗；後續已用上面的 W&B 目錄設定解決。最終模型另行保存，不能因 pilot 的 artifact 錯誤聲稱訓練未執行或所有模型已上傳。

## 評分

立即觀看一場遊戲（藍色為模型，紅色為對手）：

```powershell
python watch_astro_party.py --opponent sniper --seed 0
python watch_astro_party.py --opponent rusher --seed 0 --games 5
```

原版評分加 `--render` 只顯示對每個 baseline 的最後一場，必須先等前面的測試完成；`watch_astro_party.py` 則直接顯示指定的對戰。兩者使用相同的原版 renderer。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
../.venv311/Scripts/python.exe eval_astro_party.py --model models/model_astro_party.zip
../.venv311/Scripts/python.exe eval_astro_party.py --model models/model_astro_party.zip --seed-base 50000
```

公開通過門檻為 Rusher 90 / 100、Sniper 85 / 100。保留種子是本地泛化測試，不是助教的真正 private test，也不能保證 round robin 排名。

`experiments/final_eval.py` 另外呼叫未改動的官方 `load_model` 和 `play`，輸出逐局 JSON、Wilson 95% 區間、模型 SHA-256 與 W&B table。

## 檔案與提交範圍

原始評分器、遊戲邏輯與兩個 baseline 保持不變；雜湊清單見 `results/protected_hashes.json`。`original/` 保存本次修改前的兩個訓練檔。

正式提交 Astro Party 部分需使用 `train_astro_party.py`、`envs/train_astro_party_env.py`、`models/model_astro_party.zip`，以及合併後的作業報告。不要把虛擬環境、W&B 快取、診斷實驗或所有 checkpoint 塞進提交壓縮檔。本次沒有完成 2048，亦未提供學號，所以不冒充一份完整的全作業繳交包。

LLM 協作須依課程規定揭露；本次程式、實驗與報告有 AI 協助。


## 高速掩體實驗：速度獎勵、同向繞行與追撞

此實驗已完成 40,960 步 PPO，獨立 checkpoint 為 `models/model_astro_party_fast_cover.zip`。原正式模型 `models/model_astro_party.zip` 保留，未被較弱的實驗覆蓋。

```powershell
python train_astro_party.py --controller fast-cover
python watch_astro_party.py --model models/model_astro_party_fast_cover.zip --opponent sniper --seed 0
```

`train_astro_party_env.py` 的 fast_cover reward 增加速度 bonus：`0.0015 * min(有效速度 / 260, 1) * 情境權重`。對方仍是船時，中央行星遮蔽下權重 1、暴露時 0.25；對方變小人時只獎勵朝對方移動的速度，approach coefficient 提高至 6；自身出艙時不發速度 bonus。每步最大 bonus 小於時間成本，避免單靠繞圈取得正回報。

控制器估計對方繞中央行星的角速度，優先同向快速移動。有一發且冷卻與前一發飛行窗口結束即可探頭；敵人出艙時優先追撞，仍保留原本的 60 步避彈搜尋。高速移動、瞄準、近距離防禦會互相競爭，並不保證每幀同向或所有缺彈時間都能躲好。

固定種子 65000–65099，每個版本對各 baseline 各 100 場：

| 版本 | Rusher 勝率 | Sniper 勝率 | 對 Sniper 平均船速 | Sniper 平手 |
| --- | --- | --- | --- | --- |
| 舊掩體 | 86% | 76% | 128.5 px/s | 13% |
| 高速、無速度 bonus | 72% | 77% | 168.1 px/s | 1% |
| 高速、有速度 bonus | 72% | 80% | 166.2 px/s | 1% |

速度提升主要來自新導航；此單一訓練 seed 的對照沒有顯示速度 bonus 額外提高船速。對 Sniper 退避期間遮蔽從 32% 提升至 43%，但真正零彈期間遮蔽僅約 3%，尚未實現穩定的缺彈藏身。撞擊獲勝比例增加，但對方出艙後的完成率沒有改善。這版未達兩個 baseline 門檻，因此不取代正式模型。完整比較見 `FAST_COVER_EXPERIMENT_TW.md`、`results/fast_cover_comparison.json`。

已驗證的安全與重現性：實際訓練 checkpoint 在 180 個避彈介入狀態都提高模型預測的最小子彈距離（中位改善 11.44 px；這不是存活保證）；固定零彈、無自身子彈的近距離小人情境在 46 步／0.92 秒完成追撞。原 reward profile 與舊版 1,000 步回歸一致。最新訓練檔在乾淨平面繳交目錄執行裸命令，40,960 步後所有 policy tensors 與正式模型完全相同（最大差 0）。驗證環境為目前鎖定的 Python 3.11 套件版本。

W&B 已完成並確認雲端資料，新增可視化欄位為 `motion/ship_speed_mean`、`motion/velocity_reward_mean`、`motion/covered_mean`、`motion/hunt_mode_mean`，橫軸選 `global_step`。

- [高速有速度獎勵訓練](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/xdfwkau6)
- [相同設定、零速度獎勵對照](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/4r0t3h22)
- [同種子實際 checkpoint 比較](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/x4hdfhjo)
- [最新裸命令正式模型重現](https://forge.coreweave.com/wandb/kevinboyulien-national-taiwan-university/astro-party-hw3/runs/ghme0u0w)

關閉速度獎勵的對照命令：

```powershell
python train_astro_party.py --controller fast-cover --velocity-coef 0 --save-path models/model_astro_party_fast_cover_novelo
```

裸命令 `python train_astro_party.py` 仍重現正式通過版本；`--controller fast-cover` 才選這次實驗。W&B 預設記錄訓練數值與設定，不自動上傳模型或報告。
