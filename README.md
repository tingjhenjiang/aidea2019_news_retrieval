# AI Cup 2019 新聞立場檢索技術獎金賽
## 簡介

AIDEA 2019教育部人工智慧競賽新聞立場檢索技術獎金賽（[競賽網頁連結](https://aidea-web.tw/topic/b6abbf14-2d60-456c-8cbe-34fdfcd58967)）
主軸是包含資訊檢索、推薦系統實現。

主辦單位的描述如下：
> 開發一搜尋引擎，找出「與爭議性議題相關」且「符合特定立場」的新聞。本競賽網站以網頁連結（Hyperlink）方式，提供國內各大媒體新聞作為競賽用的資料；本網站亦提供參賽隊伍一些「包含立場和爭議性議題」的查詢題目（例如：「反對學雜費調漲」）以及部分標註資料（例如：「相關」與「不相關」），協助參賽隊伍應用「資訊檢索」及「機器學習」技術於檢索模型的訓練，期望所開發之搜尋引擎能有效找出與「反對學雜費調漲」的相關新聞，並依照相關程度由高至低排列。

訓練資料範例如下：

| Query  | News_Index  | Relevance  |
| ------------ | ------------ | ------------ |
| 支持陳前總統保外就醫  | news_064209  | 2  |
|  贊同課綱微調 | news_011362  | 1  |

## 解題作法／機器學習建模與訓練

- 資料擴增（原來的訓練資料僅有4743筆，最後訓練資料可以達到至少522,810筆資料以上）：
    - 透過同義字、反義字代換增加訓練資料
    - 將某一query-news的正例資料，改news變成負例資料
- 透過class weight->sample weight處理資料不平衡的問題
- 使用雙塔模型訓練學習 representation（全部有4,338,438 parameters）：
    - Transformer（tiny Albert, from [ckip-transformers](https://github.com/ckiplab/ckip-transformers)）
    - fully connected layers(l2 regularization)
    - dropout
    - cosine similarity
    - classifier
    - 累積機率的 loss function, from [condor-tensorflow](https://garrettjenkinson.github.io/condor_tensorflow/)
- 計畫學習率（warmup + cosine decay + restart）
- 停滯時降低學習率
- Google Colab TPU增加訓練速度

## 解題作法／曾經嘗試
- 前後嘗試不同的預訓練模型、dropout rate、fully-connected layer層數與units
- 凍結不凍結transformer的參數（不凍結，train from scratch成效較佳）
- 使用Azure多計算節點平行訓練（結果發現學生版最多只能用一個T4計算節點）

## 解題作法／系統建置部分
以Flask實施簡單的前後端介面並介接機器學習模型。系統啟動時先將所有新聞資料轉換為embedding，將所有embedding透過faiss套件建立快取索引。接著Flask的Web Server開始運作，前端方面建立查詢表單，使用者使用者可以送出查詢文字。使用者送出查詢文字後後端透過模型轉換為embedding，然後運用faiss實現檢索與推薦兩階段作業：
- Retrieval：透過faiss套件找尋與查詢比較cosine similarity最高的新聞
- Ranking：最後把查詢和新聞文件送入模型預測排序分數

## 成果展示
[影片連結](https://youtu.be/gjCyIXt1WUs)