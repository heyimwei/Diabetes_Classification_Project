# 糖尿病分類專案 (Diabetes Classification Project)

這個專案使用 **Pima Indians Diabetes Database** 來預測一個人是否罹患糖尿病，模型主要使用 **隨機森林 (Random Forest) 分類器**。在完成初步訓練與測試後，模型在驗證集上達到了約 **76.42%** 的準確率。

---

## 目錄
1. [專案背景](#專案背景)  
2. [功能特色](#功能特色)  
3. [資料集說明](#資料集說明)  
4. [前置準備 (環境與套件)](#前置準備環境與套件)  
5. [資料前處理](#資料前處理)  
6. [模型訓練流程](#模型訓練流程)  
7. [評估指標](#評估指標)  
8. [如何執行](#如何執行)  
9. [專案結構](#專案結構)  
10. [未來改進方向](#未來改進方向)  
11. [參考資料](#參考資料)

---

## 專案背景
在醫療領域中，糖尿病的早期診斷與預防至關重要。本專案透過機器學習的方法，利用病患的身體測量指標及血液檢測數據來判斷是否有糖尿病。  
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
```bash
pip install -r requirements.txt
