import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowRight, ArrowUpRight, BarChart3, Check, CheckCircle2, ChevronRight, Clock3, Copy, Cpu, FileText, Fingerprint, FlaskConical, Info, Languages, LoaderCircle, LockKeyhole, Mail, ScanLine, ShieldCheck, Sparkles, TriangleAlert, X } from 'lucide-react'
import './styles.css'

type Locale = 'zh' | 'en'
type Label = 'normal' | 'spam' | 'scam'
type Evidence = { start: number; end: number; text: string; rule: string; reason: Record<Locale, string> }
type Result = { model: 'lr' | 'bert'; label: Label; confidence: number; probabilities: Record<Label, number>; scam_type: string | null; scam_type_confidence: number | null; evidence: Evidence[]; low_confidence: boolean; truncated: boolean; latency_ms: number }
type Analysis = { results: Result[]; errors: { model: string; code: string; message: string }[]; disagreement: boolean }
type Health = { models: Record<'lr' | 'bert', boolean>; scam_types: Record<string, Record<Locale, string>> }
type Metrics = Record<string, { test_n: number; accuracy: number; macro_f1: number; metadata: { train_n?: number; frozen_encoder_layers?: number }; groups: { source: Record<string, { n: number; macro_f1: number }> } }>
const labels: Record<Locale, Record<Label, string>> = { zh: { normal: '正常訊息', spam: '垃圾訊息', scam: '疑似詐騙' }, en: { normal: 'Normal', spam: 'Spam', scam: 'Potential scam' } }
const samples = {
  zh: [
    ['投資邀約', '老師帶單保證獲利，每日收益20%，立即加入投資群組並匯款30000元開通。請不要告訴銀行，今天名額有限。'],
    ['帳號通知', '您的銀行帳號將於24小時內停用。請立即前往 https://secure-verify.example/login 輸入密碼和一次性驗證碼，以完成帳戶驗證。'],
    ['正常訊息', '老師您好，附件是我們的期中專案報告草稿。下週三下午兩點開會，想請教資料分析的方法，謝謝！'],
    ['促銷廣告', '本週會員購物優惠，全館八折，消費滿1200元免運。歡迎到門市選購！不想收到此類促銷訊息，可回覆退訂。'],
  ],
  en: [
    ['Investment', 'Our trading expert guarantees 20% daily returns. Join the investment group and wire 30000 to activate your account. Do not tell your bank. Limited places today!'],
    ['Account alert', 'Your bank account will be suspended within 24 hours. Visit https://secure-verify.example/login immediately and enter your password and one-time verification code.'],
    ['Normal email', 'Hi Professor, attached is the draft of our midterm project report. Our meeting is next Wednesday at 2 pm. We would appreciate your advice on the data analysis. Thank you!'],
    ['Promotion', 'Member sale this week: 20% off in store and free shipping on orders over 1200. Visit our shop to browse! Reply STOP to unsubscribe from promotional messages.'],
  ],
}

const pct = (n: number) => `${(n * 100).toFixed(1)}%`
const tx = (locale: Locale, zh: string, en: string) => locale === 'zh' ? zh : en

function HighlightedText({ text, spans }: { text: string; spans: Evidence[] }) {
  const points = Array.from(text)
  const parts: React.ReactNode[] = []
  let cursor = 0
  spans.forEach((span, i) => {
    parts.push(points.slice(cursor, span.start).join(''))
    parts.push(<mark key={i}>{points.slice(span.start, span.end).join('')}</mark>)
    cursor = span.end
  })
  parts.push(points.slice(cursor).join(''))
  return <div className="highlight-text">{parts}</div>
}

function App() {
  const [locale, setLocale] = useState<Locale>('zh')
  const [text, setText] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [analyzedText, setAnalyzedText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<'compare' | 'lr' | 'bert'>('compare')
  const [evidenceModel, setEvidenceModel] = useState('lr')
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [showMetrics, setShowMetrics] = useState(false)
  const [copied, setCopied] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)
  const t = (zh: string, en: string) => tx(locale, zh, en)

  useEffect(() => {
    document.documentElement.lang = locale === 'zh' ? 'zh-Hant' : 'en'
    document.title = t('MailGuard · 訊息風險分析', 'MailGuard · Message risk analysis')
  }, [locale])
  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/health', { signal: controller.signal }).then(r => { if (!r.ok) throw Error(); return r.json() }).then(setHealth).catch(() => {})
    return () => { controller.abort(); abortRef.current?.abort() }
  }, [])

  function changeText(value: string) {
    abortRef.current?.abort()
    setText(value); setAnalysis(null); setError(''); setLoading(false)
  }

  async function analyze() {
    if (text.trim().length < 3 || loading) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true); setError(''); setAnalysis(null)
    const input = text
    try {
      const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: input, models: selected === 'compare' ? ['lr', 'bert'] : [selected] }), signal: controller.signal })
      if (!response.ok) {
        if (response.status === 503) throw new Error(t('選取的模型尚未就緒。請確認訓練已完成，並已放入模型檔。', 'The selected models are not ready. Complete training and install the model artifacts.'))
        if (response.status === 422) throw new Error(t('請輸入 3 到 12,000 字的有效文字。', 'Enter valid text between 3 and 12,000 characters.'))
        throw new Error(t('分析暫時無法完成，請稍後重試。', 'Analysis could not complete. Please try again.'))
      }
      const result: Analysis = await response.json()
      if (controller.signal.aborted) return
      setAnalysis(result); setAnalyzedText(input)
      setEvidenceModel(result.results.find(r => r.label === 'scam')?.model || result.results[0].model)
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
    } catch (e) {
      if (!controller.signal.aborted) setError(e instanceof Error ? e.message : t('無法連線至服務，請確認後端正在執行。', 'Cannot connect to the service. Check that the backend is running.'))
    } finally {
      if (!controller.signal.aborted) setLoading(false)
    }
  }

  async function openMetrics() {
    setShowMetrics(true)
    try { const r = await fetch('/api/metrics'); if (r.ok) setMetrics(await r.json()) } catch { setMetrics({}) }
  }
  const active = analysis?.results.find(r => r.model === evidenceModel)
  const scamResults = analysis?.results.filter(r => r.label === 'scam') ?? []

  return <>
    <header className="topbar"><div className="topbar-inner">
      <a className="brand" href="#" aria-label="MailGuard"><span className="brand-icon"><ShieldCheck size={23} strokeWidth={1.8}/></span><span>Mail<span className="brand-light">Guard</span><small>MESSAGE INTELLIGENCE</small></span></a>
      <nav><button className="nav-link" onClick={openMetrics}>{t('模型與評估', 'Models & evaluation')}<ArrowUpRight size={14}/></button><span className="nav-divider"/><button className="language-button" onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}><Languages size={16}/>{locale === 'zh' ? 'English' : '繁體中文'}</button></nav>
    </div></header>

    <main>
      <section className="hero">
        <div className="eyebrow"><span className="status-dot"/>{t('雙模型・雙語訊息分析', 'TWO MODELS. TWO LANGUAGES. ONE CLEARER PICTURE.')}</div>
        <h1>{t('讓可疑訊息，', 'A second look.')}<span>{t('有跡可循。', 'A clearer inbox.')}</span></h1>
        <p>{t('從一封郵件、一則簡訊開始。辨識訊息類別、找出可疑線索，讓每一次點擊都有多一分把握。', 'From an email to a text message. Compare classifications and inspect suspicious phrases before deciding what to do next.')}</p>
        <div className="hero-chips"><span><Languages size={14}/>{t('繁體中文 / English', 'English / Traditional Chinese')}</span><span><LockKeyhole size={14}/>{t('輸入內容不留存', 'Your input is not stored')}</span><span><FlaskConical size={14}/>{t('課堂研究展示', 'Research demo')}</span></div>
        <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="art-envelope"><Mail size={62} strokeWidth={1}/><span className="art-shield"><ShieldCheck size={31} strokeWidth={1.6}/></span></div><span className="art-spark spark-one">+</span><span className="art-spark spark-two">+</span><span className="art-node node-one"/><span className="art-node node-two"/></div>
      </section>

      <div className="workspace">
        <section className="editor-card" aria-labelledby="editor-title">
          <div className="card-heading"><div><span className="step-number">01</span><h2 id="editor-title">{t('貼上你的訊息', 'Paste your message')}</h2></div><span className="small-pill">{t('郵件 / 簡訊 / 文字', 'EMAIL / SMS / TEXT')}</span></div>
          <div className="sample-row"><span>{t('試試範例', 'Try an example')}</span>{samples[locale].map(([label, value]) => <button key={label} onClick={() => changeText(value)}>{label}<ArrowUpRight size={12}/></button>)}</div>
          <div className="textarea-wrap"><textarea aria-label={t('待分析的郵件或文字', 'Email or text to analyze')} placeholder={t('在這裡貼上電子郵件或任何讓你感到可疑的訊息⋯\n\n可包含主旨與內文，請先移除不必要的個人資料。', 'Paste an email or any message that makes you pause…\n\nYou can include the subject and body. Remove unnecessary personal information first.')} value={text} onChange={e => changeText(e.target.value)} maxLength={12000} onKeyDown={e => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); analyze() } }}/><div className="textarea-footer"><span><LockKeyhole size={12}/>{t('僅送至本機服務進行分析', 'Analyzed by your local service')}</span><span>{Array.from(text).length.toLocaleString()} / 12,000</span></div></div>
          <div className="model-selection"><span className="field-label">{t('分析模型', 'ANALYSIS MODEL')}</span><div className="model-tabs" role="group" aria-label={t('選擇模型', 'Select a model')}>{(['compare', 'lr', 'bert'] as const).map(name => <button key={name} className={selected === name ? 'selected' : ''} aria-pressed={selected === name} disabled={loading} onClick={() => { setSelected(name); setAnalysis(null) }}>{name === 'compare' ? <BarChart3 size={15}/> : <Cpu size={15}/>}{name === 'compare' ? t('雙模型比較', 'Compare both') : name === 'lr' ? 'Logistic Regression' : 'BERT'}</button>)}</div></div>
          {error && <div className="error-message" role="alert"><TriangleAlert size={17}/>{error}</div>}
          <div className="editor-actions"><span className="keyboard-hint">⌘ / Ctrl + Enter</span><button className="analyze-button" onClick={analyze} disabled={text.trim().length < 3 || loading}>{loading ? <><LoaderCircle className="spin" size={18}/>{t('正在分析訊息⋯', 'Analyzing…')}</> : <><ScanLine size={18}/>{t('開始分析', 'Analyze message')}<ArrowRight size={18}/></>}</button></div>
        </section>

        <aside className="guide-card"><div className="guide-icon"><Fingerprint size={28} strokeWidth={1.4}/></div><h2>{t('不只分類，', 'Beyond a label.')}<br/>{t('也看懂線索。', 'Understand the signals.')}</h2><p>{t('兩種不同方法，提供更多觀察角度。', 'Two different approaches for a more informed perspective.')}</p><div className="guide-step"><span>1</span><div><h3>{t('辨識訊息類別', 'Classify the message')}</h3><p>{t('正常、垃圾或疑似詐騙，呈現各類別機率。', 'Normal, spam, or scam, with all three probabilities.')}</p></div></div><div className="guide-step"><span>2</span><div><h3>{t('檢視可疑文字', 'Inspect suspicious phrases')}</h3><p>{t('用可追溯的規則標示風險線索，說明原因。', 'Traceable rules highlight risk cues and explain why.')}</p></div></div><div className="guide-step"><span>3</span><div><h3>{t('了解詐騙類型', 'Identify the scam category')}</h3><p>{t('從投資到帳號盜用，對應六大類型。', 'Six categories, from investment fraud to account theft.')}</p></div></div><div className="guide-note"><Info size={15}/><span>{t('機率是模型信心，不能保證訊息安全。', 'Probabilities reflect model confidence, not guaranteed safety.')}</span></div></aside>
      </div>

      <section className="results-section" ref={resultsRef} aria-live="polite" aria-busy={loading}>
        <div className="section-heading"><div><span className="step-number">02</span><h2>{t('分析結果', 'Analysis results')}</h2></div><span>{analysis ? t('根據這次輸入的訊息', 'Based on your current message') : t('你的下一個判斷，從這裡開始', 'Your next decision starts here')}</span></div>
        {loading ? <div className="empty-state"><LoaderCircle className="spin" size={32}/><h3>{t('模型正在閱讀你的訊息', 'The models are reading your message')}</h3><p>{t('BERT 在 CPU 上第一次載入可能需要較長時間，請稍候。', 'The first BERT request may take longer on a CPU. Please wait.')}</p></div> : !analysis ? <div className="empty-state"><div className="empty-icon"><ScanLine size={25}/></div><h3>{t('多一個檢查，少一分疑慮', 'A little clarity goes a long way')}</h3><p>{t('貼上訊息並開始分析，這裡將顯示分類、信心分數與可疑線索。', 'Paste a message and start an analysis to see classifications, confidence scores, and risk cues.')}</p><div className="legend"><span><i className="dot-normal"/>{labels[locale].normal}</span><span><i className="dot-spam"/>{labels[locale].spam}</span><span><i className="dot-scam"/>{labels[locale].scam}</span></div></div> : <>
          {analysis.disagreement && <div className="notice warning"><TriangleAlert size={18}/>{t('兩個模型的分類不同。請比較機率與線索，並透過官方管道核實內容。', 'The models disagree. Compare their probabilities and evidence, then verify through official channels.')}</div>}
          {analysis.errors.map(e => <div key={e.model} className="notice warning"><Info size={17}/>{e.model.toUpperCase()} {t('尚未就緒或無法執行；以下僅顯示成功執行的模型。', 'is unavailable. Only successfully executed models are shown below.')}</div>)}
          <div className={`result-grid ${analysis.results.length === 1 ? 'single' : ''}`}>{analysis.results.map(r => <article key={r.model} className={`result-card ${r.label}`}><div className="result-top"><span className="model-label"><Cpu size={16}/>{r.model === 'lr' ? 'Logistic Regression' : 'Multilingual BERT'}</span><span className="latency"><Clock3 size={12}/>{r.latency_ms >= 1000 ? `${(r.latency_ms/1000).toFixed(1)}s` : `${r.latency_ms}ms`}</span></div><div className="prediction"><div><span className="result-tag">{r.label === 'normal' ? <CheckCircle2 size={15}/> : <TriangleAlert size={15}/>} {labels[locale][r.label]}</span><h3>{pct(r.confidence)}</h3><span className="confidence-label">{t('最高類別機率', 'HIGHEST CLASS PROBABILITY')}</span></div><div className={`confidence-ring ${r.label}`} style={{ '--progress': `${r.confidence * 360}deg` } as React.CSSProperties}><span>{r.label === 'normal' ? <ShieldCheck size={27}/> : r.label === 'scam' ? <TriangleAlert size={27}/> : <Mail size={27}/>}</span></div></div><div className="probability-list">{(['normal', 'spam', 'scam'] as Label[]).map(label => <div className="probability-row" key={label}><span>{labels[locale][label]}</span><div className="bar-track"><div className={`bar-fill ${label}`} style={{ width: `${r.probabilities[label]*100}%` }}/></div><strong>{pct(r.probabilities[label])}</strong></div>)}</div>{r.scam_type && <div className="scam-type"><span>{t('推測詐騙類型', 'PREDICTED SCAM CATEGORY')}</span><strong>{health?.scam_types[r.scam_type]?.[locale] ?? r.scam_type}</strong><small>{t('類型信心（未校準）', 'Type confidence (uncalibrated)')} {pct(r.scam_type_confidence ?? 0)}</small></div>}{r.low_confidence && <div className="inline-warning"><Info size={14}/>{t('模型信心較低，建議人工核實。', 'Low confidence. Manual verification is recommended.')}</div>}{r.truncated && <div className="inline-warning"><Info size={14}/>{t('內容較長，BERT 僅分析開頭的 token 範圍，後段內容未納入分類。', 'BERT analyzed only the first token window. Later content was not classified.')}</div>}</article>)}</div>
          {scamResults.length > 0 && <section className="evidence-card"><div className="evidence-heading"><div><span className="step-number">03</span><h2>{t('可疑線索與說明', 'Suspicious phrases & context')}</h2></div><div className="small-tabs">{scamResults.map(r => <button key={r.model} className={evidenceModel === r.model ? 'active' : ''} onClick={() => setEvidenceModel(r.model)}>{r.model === 'lr' ? 'LR' : 'BERT'}</button>)}</div></div><p className="evidence-caption">{t('以下標記由透明規則產生，輔助閱讀；不是模型的注意力或因果解釋。', 'Highlights come from transparent evidence rules. They are reading aids, not model attention or causal explanations.')}</p><div className="evidence-layout"><HighlightedText text={analyzedText} spans={active?.evidence ?? []}/><div className="evidence-reasons">{active?.evidence.length ? active.evidence.map((span, i) => <div className="evidence-reason" key={i}><span className="evidence-index">{String(i+1).padStart(2, '0')}</span><div><strong>{span.text}</strong><p>{span.reason[locale]}</p></div></div>) : <div className="no-evidence"><Info size={18}/><p>{t('模型判為疑似詐騙，但規則未找到可標示的文字。這不代表沒有風險，請人工核實。', 'The model predicted a scam, but no rule matched. This does not establish safety; verify manually.')}</p></div>}</div></div></section>}
          <div className="result-footer"><span><LockKeyhole size={13}/>{t('這次分析未留存輸入內容', 'Your input was not retained')}</span><button onClick={async () => { try { await navigator.clipboard.writeText(JSON.stringify({ ...analysis, input: analyzedText }, null, 2)); setCopied(true); setTimeout(() => setCopied(false), 1800) } catch { setError(t('無法使用剪貼簿，請檢查瀏覽器權限。', 'Clipboard unavailable. Check browser permissions.')) } }}>{copied ? <Check size={14}/> : <Copy size={14}/>}{copied ? t('已複製', 'Copied') : t('複製分析結果', 'Copy analysis')}</button></div>
        </>}
      </section>
      <section className="method-strip"><div><span className="method-icon"><FlaskConical size={20}/></span><div><h3>{t('看得見的方法，值得追問的結果', 'Transparent methods. Results worth questioning.')}</h3><p>{t('TF-IDF + Logistic Regression 與多語言 BERT，使用相同獨立測試集評估；資料來源與限制公開記錄。', 'TF-IDF + Logistic Regression and multilingual BERT, evaluated on the same held-out split with documented sources and limitations.')}</p></div></div><button onClick={openMetrics}>{t('查看模型評估', 'View evaluation')}<ArrowUpRight size={16}/></button></section>
    </main>
    <footer><div><span className="footer-brand"><ShieldCheck size={17}/>MailGuard</span><span>MMIP · MIDTERM PROJECT</span></div><p>{t('研究展示工具。涉及付款、帳密或人身安全時，請向官方機構查證。', 'Research demonstration. Verify requests for money, credentials, or personal safety with official sources.')}</p></footer>

    {showMetrics && <div className="modal-backdrop" onClick={() => setShowMetrics(false)}><section className="metrics-modal" role="dialog" aria-modal="true" aria-labelledby="metrics-title" onClick={e => e.stopPropagation()}><button className="close-modal" aria-label={t('關閉', 'Close')} onClick={() => setShowMetrics(false)}><X size={20}/></button><span className="eyebrow"><FlaskConical size={14}/> MODEL CARD</span><h2 id="metrics-title">{t('誠實看待每一個分數', 'Every score needs context.')}</h2><p>{t('目前資料含公開英文語料與繁中／英文合成範例。六種詐騙類型主要依賴合成資料及弱標註，不能將測試分數當作真實世界準確率。', 'The dataset combines public English corpora with synthetic Chinese/English examples. Six-category supervision relies primarily on synthetic and weak labels. Scores are not real-world accuracy claims.')}</p>{metrics === null ? <LoaderCircle className="spin"/> : Object.keys(metrics).length === 0 ? <div className="notice">{t('目前尚無已完成的模型評估。', 'No completed model evaluations are available yet.')}</div> : Object.entries(metrics).map(([name, m]) => <div className="metric-model" key={name}><h3>{name === 'lr' ? 'Logistic Regression' : 'Multilingual BERT'}</h3><div className="metric-numbers"><div><strong>{m.macro_f1.toFixed(3)}</strong><span>Macro-F1</span></div><div><strong>{pct(m.accuracy)}</strong><span>{t('測試準確率', 'Test accuracy')}</span></div><div><strong>{m.test_n.toLocaleString()}</strong><span>{t('測試筆數', 'Test examples')}</span></div></div>{name === 'bert' && <p>{t('BERT 訓練筆數', 'BERT training examples')}：{m.metadata.train_n?.toLocaleString()} · {t('凍結編碼層數', 'Frozen encoder layers')}：{m.metadata.frozen_encoder_layers}</p>}<div className="source-scores">{Object.entries(m.groups.source).map(([source, v]) => <div key={source}><span>{source === 'bilingual_templates' ? t('合成範例', 'Synthetic examples') : source}</span><span>n={v.n} · F1 {v.macro_f1.toFixed(3)}</span></div>)}</div></div>)}<div className="notice"><Info size={16}/>{t('主分類機率經驗證集校準；六類型機率未校準。可疑文字來自獨立規則。簡體輸入先轉繁體，但標記保留原文。', 'Main probabilities are calibrated on validation data; subtype probabilities are uncalibrated. Evidence uses independent rules. Simplified input is normalized to Traditional Chinese while highlights preserve the original text.')}</div></section></div>}
  </>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>)
