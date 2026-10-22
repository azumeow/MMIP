"""Transparent, versioned evidence rules, independent of model classification.

These spans are evidence cues, NOT BERT attention, SHAP, or proof of fraud.
All offsets refer to the original user input (including Simplified Chinese).
"""
import re
from ml.common import converter

RULES = [
    ('guaranteed_return', 'investment', r'保[證证][^。.!！\n]{0,12}(?:獲利|收益|回報|回报|報酬|获利)|穩賺不賠|稳赚不赔|零風險|零风险|guaranteed\s+(?:profit|returns?)|risk[- ]free\s+(?:profit|investment)', '承諾保證獲利或無風險報酬', 'Promises guaranteed returns or risk-free investing'),
    ('investment_platform', 'investment', r'投資群組|投资群组|帶單|带单|內線消息|内幕消息|虛擬貨幣|虚拟货币|crypto(?:currency)?|investment\s+(?:group|platform)|insider\s+(?:tip|information)', '投資或帶單邀約，需要核實平台與資格', 'Investment solicitation; verify the platform and credentials'),
    ('shopping_payment', 'shopping', r'私下交易|私下匯款|私下汇款|解除分期|重複扣款|重复扣款|貨到付款|货到付款|off[- ]platform|duplicate\s+charge|cancel\s+installments?|courier\s+fee|redelivery\s+fee', '要求平台外付款或以訂單問題催付', 'Requests off-platform payment or payment for an order problem'),
    ('social_money', 'relationship', r'親愛的[^。.!！\n]{0,45}(?:匯款|轉帳|转账|汇款)|寶貝[^。.!！\n]{0,45}(?:急|錢|钱)|猜猜我是誰|猜猜我是谁|我是你(?:的)?(?:朋友|同學|同学)|my\s+love[^.!\n]{0,70}(?:send|money|transfer)|stranded\s+(?:abroad|overseas)|romance|gift\s+cards?', '透過親密關係、身分冒充或禮物卡要求金錢', 'Uses personal trust, impersonation, or gift cards to request money'),
    ('government_identity', 'government', r'檢察官|检察官|地檢署|地检署|法院|警察|公安|監管帳戶|监管账户|安全帳戶|安全账户|稅務局|税务局|prosecutor|police|\bIRS\b|tax\s+authority|government\s+grant|court\s+order', '聲稱政府或執法人員，應使用官方管道核實', 'Claims government or law-enforcement authority; verify independently'),
    ('job_loan_advance', 'job_loan', r'入職[^。.!！\n]{0,15}(?:費|费|押金)|入职[^。.!！\n]{0,15}(?:費|费|押金)|(?:貸款|贷款)[^。.!！\n]{0,25}(?:先付|保證金|保证金|手續費|手续费)|刷單|刷单|高薪[^。.!！\n]{0,15}(?:兼職|兼职)|processing\s+fee|upfront\s+(?:fee|deposit)|loan\s+approved|pay\s+to\s+(?:start|work)|task\s+(?:job|commission)', '以工作或貸款為由先收費', 'Asks for advance fees for a job or loan'),
    ('credential_request', 'account', r'驗證碼|验证码|一次性密碼|一次性密码|助記詞|助记词|私鑰|私钥|one[- ]time\s+(?:password|code)|\bOTP\b|seed\s+phrase|private\s+key|(?:enter|send|share|confirm)[^.!\n]{0,35}(?:password|credentials)', '要求驗證碼、密碼或金鑰，屬敏感資訊', 'Requests a verification code, password, or secret key'),
    ('account_threat', 'account', r'[帳賬账][號号户戶][^。.!！\n]{0,20}(?:停用|凍結|冻结|封鎖|封锁|異常|异常)|account[^.!\n]{0,35}(?:suspend|locked|disabled|unusual)|remote\s+(?:access|support)|遠端控制|远程控制', '以帳號異常或遠端協助引導操作', 'Uses account threats or remote access to prompt action'),
    ('urgency', None, r'立即|立刻|限時|限时|最後通知|最后通知|\b(?:urgent|immediately|final\s+notice|within\s+24\s+hours)\b', '製造急迫感；單獨出現並不代表詐騙', 'Creates urgency; this cue alone does not establish fraud'),
    ('money_transfer', None, r'匯款|汇款|轉帳|转账|銀行帳號|银行账号|wire\s+transfer|send\s+money|transfer\s+(?:money|funds)|pay\s+(?:now|immediately)', '涉及付款或匯款，請先核實收款對象', 'Requests payment or transfer; verify the recipient'),
    ('external_url', None, r'https?://[^\s<>"\u3000]+', '包含外部連結；系統不會開啟或驗證連結安全性', 'Contains an external URL; links are not opened or checked'),
]

def evidence(text: str) -> list[dict]:
    spans = []
    for key, subtype, pattern, zh, en in RULES:
        for match in re.finditer(pattern, text, flags=re.I):
            spans.append({'start': match.start(), 'end': match.end(), 'text': match.group(), 'rule': key, 'scam_type': subtype, 'reason': {'zh': zh, 'en': en}})
    # Keep non-overlapping spans with deterministic priority for longer matches.
    selected = []
    for span in sorted(spans, key=lambda s: (s['start'], -(s['end']-s['start']))):
        if not selected or span['start'] >= selected[-1]['end']:
            selected.append(span)
    return selected[:60]

def weak_subtype(text: str) -> str | None:
    counts = {}
    for span in evidence(converter().convert(text)):
        if span['scam_type']:
            counts[span['scam_type']] = counts.get(span['scam_type'], 0) + 1
    return max(counts, key=counts.get) if counts else None
