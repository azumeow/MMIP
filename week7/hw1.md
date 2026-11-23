# 作業一：Router 與 Scheme 設計

| 檔案 | 內容 |
|---|---|
| [router.json](hw1/router.json) | 6 條 route 的說明：returns、customer_support、out_of_scope、place_order、order_tracking、clarification |
| [scheme.json](hw1/scheme.json) | 完整業務規格：`router`、商品型錄、業務規則、每條 route 的必填 slots、回覆模板、`router_output_spec`（Router 輸出 JSON Schema）、多意圖輸出格式 |
| [架構圖.md](hw1/架構圖.md) | 依上述兩份檔案繪製的系統流程圖（Mermaid） |

作業二、三的程式都直接讀取 `hw1/scheme.json`，所以修改這份檔案會影響後續的 Router 提示詞與客服回覆。
