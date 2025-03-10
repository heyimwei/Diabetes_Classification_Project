# 糖尿病分類專案 (Diabetes Classification Project)

這個專案使用 **Pima Indians Diabetes Database** 資料集來訓練模型，預測一個人是否罹患糖尿病，模型主要使用 **隨機森林 (Random Forest) 分類器**。在完成訓練與測試後，模型在驗證集上達到了約 **76.42%** 的準確率。

---

## 目錄
1. [專案背景](#專案背景)  
2. [功能特色](#功能特色)  
3. [資料集說明](#資料集說明)  
4. [前置準備 (環境與套件)](#前置準備環境與套件)  
5. [資料前處理](#資料前處理)  
6. [模型訓練流程](#模型訓練流程)  
7. [評估指標](#評估指標)  

---

## 專案背景
在醫療領域中，糖尿病的早期診斷與預防至關重要。本專案透過機器學習的方法，利用病患的身體檢測數據來判斷是否有糖尿病。  
此專案的主旨在於示範如何：
- 進行資料前處理（例如處理遺漏值、不合理數值）  
- 利用特徵重要度判斷哪些特徵最能影響模型表現  
- 訓練並評估 **Random Forest** 模型的效能  

---

## 功能特色
1. **資料前處理**：將不合理的 0 值轉成缺失值並以平均值填補。  
2. **特徵工程**：根據隨機森林特徵重要度篩選關鍵特徵。  
3. **模型訓練**：使用 **Random Forest** 分類器進行模型訓練與預測。  
4. **評估指標**：輸出準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall)、F1 分數。  
5. **結果可視化**：透過圖表展示特徵重要度。

---

## 資料集說明
- **名稱**：Pima Indians Diabetes Database  
- **筆數**：768  
- **欄位**：8 個特徵 + 1 個目標 (0：無糖尿病，1：有糖尿病)  
- **特徵**：
  1. `Number of Times Pregnant` (懷孕次數)  
  2. `Plasma Glucose Concentration` (血糖濃度)  
  3. `Diastolic Blood Pressure` (舒張壓)  
  4. `Triceps Skin Fold Thickness` (皮膚皺褶厚度)  
  5. `2-Hour Serum Insulin` (2 小時血清胰島素)  
  6. `BMI` (身體質量指數)  
  7. `Diabetes Pedigree Function` (家族糖尿病病史函數)  
  8. `Age` (年齡)  

---

## 前置準備 (環境與套件)
在執行之前，請確保已安裝以下 Python 套件 (可使用 `requirements.txt` 進行安裝)：
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- jupyter (若需要跑 Notebook)

安裝方法：
```pip install -r requirements.txt```

---

## 資料前處理

處理 0 值：
- 在血糖濃度 (glucose)、血壓 (blood_pressure)、BMI 等醫學上不合理出現 0 值的欄位，將 0 改成使用該欄位的**平均值**進行填補。

特徵標準化 (StandardScaler)：
- 為提升訓練穩定度，將數值特徵做標準化。

---

## 模型訓練流程

資料切分：將資料分為訓練集 (train) 與驗證集 (validation)。
模型選擇：隨機森林分類器 (RandomForestClassifier)。
超參數設定：預設使用 scikit-learn 預設值。
特徵選擇：根據隨機森林的特徵重要度 (feature_importances_) 挑選最有幫助的特徵作二次訓練。
模型評估：使用驗證集，計算準確率、精確率、召回率、F1 分數。

---

## 評估指標

Accuracy：約 76.42%
Precision (0)：78%，Precision (1)：73%
Recall (0)：87%，Recall (1)：59%

模型對非糖尿病患者預測較準確，但對糖尿病患者仍有較高的漏診情況 (Recall 偏低)。