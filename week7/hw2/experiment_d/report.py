"""產生中文審核頁，明確分開原生模型回答與程式包裝。"""
import html
import json
from pathlib import Path
from classifier.common import read_rows


def write_report(output):
    """只讀取實驗紀錄；不修改機率、門檻或標籤。"""
    output=Path(output)
    rows=read_rows(output/'raw_results.jsonl')
    metrics=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    esc=lambda x:html.escape(str(x))
    cards=[]
    for row in sorted(rows,key=lambda r:r['end_to_end_success']):
        native=json.dumps(row['api_response'],ensure_ascii=False,indent=2)
        cards.append(f'''<article class="{'ok' if row['end_to_end_success'] else 'bad'}">
<h3>{esc(row['id'])} · {'通過' if row['end_to_end_success'] else '需要檢查'}</h3>
<p class="query">{esc(row['text'])}</p>
<p>正確路由：<b>{esc(row['expected_route'])}</b>；模型原始選擇：<b>{esc(row.get('native_predicted_route'))}</b>；門檻後路由：<b>{esc(row['predicted_route'])}</b></p>
<p>最大路由機率：{esc(row.get('maximum_route_probability'))}；低分拒判：{esc(row.get('rejected_by_threshold'))}；格式通過：{row['schema_valid']}；路由正確：{row['route_correct']}；耗時：{row['latency_s']:.3f} 秒</p>
<details><summary>原生模型 API 回答（選項機率）</summary><pre>{esc(native)}</pre></details>
<details><summary>程式組裝的 Router JSON</summary><pre>{esc(row['raw_output'])}</pre></details>
<pre>{esc(row['api_error'] or row['validation_error'])}</pre></article>''')
    indicators=''.join(f'<p>{label}：<b>{metrics[key]:.2%}</b></p>' for key,label in [
        ('native_route_accuracy','模型原始路由正確率'),('route_correct_rate','0.6 門檻後路由正確率'),
        ('coverage','通過機率門檻比例'),('schema_valid_rate','程式包裝後 Pydantic 通過率'),
        ('end_to_end_success_rate','格式及路由同時正確')])
    page=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>實驗 D 審核</title>
<style>body{{font-family:system-ui,"Microsoft JhengHei";background:#f1f5f9;color:#172033;line-height:1.7}}main{{max-width:1100px;margin:auto;padding:20px}}article,.intro{{background:white;padding:20px;margin:18px 0;border-radius:12px}}.bad{{border-left:5px solid #d33}}.ok{{border-left:5px solid #2a6}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f7;padding:12px}}.query{{font-size:21px}}button{{padding:12px;font:inherit}}</style>
<main><h1>實驗 D：Clef-Flash 原生 System One</h1><div class="intro">
<p>使用本機 Clef-Flash 9B Q4_K_M，並非原本的 TF-IDF 分類器。共 {len(rows)} 筆，沒有使用這些測試標籤訓練或調整門檻。</p>
<p>模型原生輸出 route / intent 選項機率，沒有逐字生成 JSON。只有最大路由機率嚴格大於 0.6 才接受，否則改為 clarification。API 的 confidence 不是這裡使用的最大機率。</p>
<p>完整 Router JSON 的 slots、status、missing_slots、next_action 由程式規則組裝；五個 slots 必定保留，不知道填 null。因此格式通過率代表介接程式的結果，不能當成模型自行生成 JSON 的能力。沒有完整 slots 人工正解，本頁不評估其語意正確率。</p>
{indicators}<p>平均耗時 {metrics['mean_latency_s']:.3f} 秒；P95 {metrics['p95_latency_s']:.3f} 秒。排除載入及一次暖機；包含 HTTP、推論及組裝。</p></div>
<button onclick="document.querySelectorAll('article.ok').forEach(e=>e.hidden=!e.hidden)">顯示／隱藏通過案例</button>
<p>需要檢查 {sum(not r['end_to_end_success'] for r in rows)} 筆，通過 {sum(r['end_to_end_success'] for r in rows)} 筆。</p>{''.join(cards)}</main>
<script>document.querySelectorAll('article.ok').forEach(e=>e.hidden=true)</script></html>'''
    (output/'report.html').write_text(page,encoding='utf-8')
