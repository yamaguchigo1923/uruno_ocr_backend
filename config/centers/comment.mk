{
  "id": "defalt",
  "displayName": "defalt",
  "templateSpreadsheetId": "1tIO4OvE5SC0NN1tDEIAbZV6xQ_4QiA3Jd7Pi2gLfN1o",
  "templateSheetId": 1557733602,
  "exportStartRow": 32,
  "coverPages": 1,

# クリップ率
  "crop": {
    "top_pct": 0.0,
    "bottom_pct": 0.0,
    "left_pct": 0.0,
    "right_pct": 0.0
  },

# 見積書作成(スプレッドシート出力)
  "poll": {
    "startCol": "B",
    "endCol": "K",
    "readyColRelativeIndex": 0,
    "minReadyRatio": 0.95,
    "maxWaitSec": 600
  },

# 探索するヘッダー列とマッチング
  "headers": {
    "judgeCandidates": ["銘柄·条件", "銘柄・条件", "銘柄条件"],
    "fallbackChars": "銘柄条件",
    "needColumns": ["成分表", "見本"]
  },

  "nutrition":{
    "module":[1],
    "col":[銘柄・条件],
    "matching":[成分表提出]
  }
  "sample":{
    "module":[2],
    "col":[備考],
    "matching_T":[◯],
    "matching_F":[-]
  }
  

# データ出力
  "ranges": {
    "catalog": "'商品'!A10:I30000",
    "export": {
      "makerHeader": "B4:E4",
      "centerName": "B27:E27",
      "month": "F27"
    }
  }
}
