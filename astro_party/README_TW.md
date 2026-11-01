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
