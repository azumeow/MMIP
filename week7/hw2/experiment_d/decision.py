"""Clef 原生 typed choices 與 RouterOutput 的介接；不讀取資料來源標籤。"""
import math
import re

from hw2.experiment_a.prompt import ROOT,make_system_prompt
from hw2.experiment_a.schemas import RouterOutput

INTENTS={
 'check_product_specs':'Ask about a particular product specification or feature.',
 'general_product_question':'General product information or usage, not an order.',
 'request_return':'Request a return, refund, reimbursement or compensation.',
 'ask_return_policy':'Ask about eligibility, conditions or policy for returns/refunds.',
 'technical_support':'Troubleshoot a product malfunction or technical issue.',
 'feedback':'Submit a complaint, review, compliment or other feedback.',
 'contact_customer_service':'Ask how or when to contact customer service.',
 'contact_human_agent':'Explicitly ask to speak with a human representative.',
 'purchase_product':'Buy a product or place a new order.',
 'cancel_order':'Cancel an existing purchase or order.',
 'check_shipping_status':'Track an order, shipment, delivery or expected arrival.',
 'unrelated_request':'A request unrelated to this electronics store.',
 'unknown':'Cannot identify the intent without more context.',
}
ROUTE_INTENTS={
 'product_inquiry':['check_product_specs','general_product_question'],
 'returns':['request_return','ask_return_policy'],
 'customer_support':['technical_support','feedback','contact_customer_service','contact_human_agent'],
 'place_order':['purchase_product','cancel_order'],
 'order_tracking':['check_shipping_status'],'out_of_scope':['unrelated_request'],'clarification':['unknown'],
}


def routing_rules():
    """取 A 同一段業務路由規則；不把 LLM 的『請生成 JSON』指示交給決策 API。"""
    return make_system_prompt().split('Routing rules:\n',1)[1].split('\nOutput rules:',1)[0].strip()


def questions(scheme):
    """兩個 choice 問題在同一次原生 forward 中共同決策，不逐字生成 JSON。"""
    return {'route':{'type':'choice',
        'instructions':'Choose the best route for the customer message.\n'+routing_rules(),
        'criteria':scheme['router']},
        'intent':{'type':'choice','instructions':'Identify the specific intent of the same customer message.',
        'criteria':INTENTS}}


def validate_choices(response,query_questions):
    """驗證所有選項都有有限機率、總和約為 1，且 choice 確為最高分之一。"""
    answers=response.get('answers',{})
    for name,question in query_questions.items():
        answer=answers.get(name,{})
        probabilities=answer.get('probabilities',{})
        if answer.get('type')!='choice' or set(probabilities)!=set(question['criteria']):
            raise ValueError(f'Invalid native choice response for {name}')
        values=list(probabilities.values())
        if any(isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1 for p in values):
            raise ValueError(f'Invalid native probabilities for {name}')
        if not math.isclose(sum(values),1,abs_tol=1e-4):
            raise ValueError(f'Probabilities do not sum to 1 for {name}')
        choice=answer.get('choice')
        if choice not in probabilities or probabilities[choice]<max(values)-1e-7:
            raise ValueError(f'Choice is not a maximum-probability option for {name}')
    return answers


def extract_slots(text,intent):
    """只抽出問題明示資訊；這是規則抽取，不使用 expected_route、來源 intent 或替換紀錄。"""
    slots=dict.fromkeys(['product_name','quantity','order_id','delivery_address','customer_name'])
    match=re.search(r'\b(?:order|purchase)\s*(?:number|no\.?|#)?\s*([0-9]{1,10})\b',text,re.I)
    if match:slots['order_id']=match.group(1)
    for pattern,name in [(r'\brtx\s*5080\b','NVIDIA GeForce RTX 5080'),
        (r'\b(?:tesla\s*)?v100\b','NVIDIA Tesla V100'),
        (r'\b(?:rx\s*)?7900\s*xtx\b','AMD Radeon RX 7900 XTX')]:
        if re.search(pattern,text,re.I):slots['product_name']=name
    if intent=='purchase_product':
        match=re.search(r'\b([1-9][0-9]*|one|two|three|four|five|six|seven|eight|nine|ten|an?)\s+(?:items?|products?|articles?|units?)\b',text,re.I)
        if match:
            value=match.group(1).lower()
            words={'a':1,'an':1,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
            slots['quantity']=int(value) if value.isdigit() else words[value]
    # 只有顯式標示 address / name 的內容才抽取，不能把機場或一般地點猜成收件地址。
    match=re.search(r'\b(?:delivery\s+|shipping\s+)?address\s*[:=]\s*([^;\n]+)',text,re.I)
    if match:slots['delivery_address']=match.group(1).strip()
    match=re.search(r'\b(?:customer\s+)?name\s*[:=]\s*([^;\n]+)',text,re.I)
    if match:slots['customer_name']=match.group(1).strip()
    return slots


def assemble(text,answers,scheme,threshold):
    """route/intent 取自模型分數；slots/status/action 由明確規則包裝成共同 JSON。"""
    route_probabilities=answers['route']['probabilities']
    native_route=answers['route']['choice']
    maximum=max(route_probabilities.values())
    rejected=maximum<=threshold  # 沿用使用者指定：必須嚴格大於 0.6 才接受。
    route='clarification' if rejected else native_route
    # 只在預測 route 的合法子意圖中選最大分；保留完整原始意圖機率供核對。
    valid_intents=ROUTE_INTENTS[route]
    intent=max(valid_intents,key=lambda name:answers['intent']['probabilities'][name])
    slots=extract_slots(text,intent)
    requirements=scheme['route_slot_requirements'][route].get(intent,[])
    if not isinstance(requirements,list):requirements=[]
    missing=[name for name in requirements if slots[name] is None]
    status,action='ready','generate_response'
    if route in ['out_of_scope','clarification']:
        slots=dict.fromkeys(slots);missing=[]
        status,action=('out_of_scope','reject_request') if route=='out_of_scope' else ('ambiguous_intent','ask_for_information')
    elif missing:status,action='missing_information','ask_for_information'
    elif intent in ['purchase_product','cancel_order']:
        status='awaiting_confirmation'
        action='request_order_confirmation' if intent=='purchase_product' else 'request_cancellation_confirmation'
    elif intent=='contact_human_agent':action='handoff_to_human'
    router={'route':route,'intent':intent,'confidence':float(maximum),'status':status,
        'extracted_slots':slots,'missing_slots':missing,'next_action':action}
    RouterOutput.model_validate(router)
    return router,{'native_predicted_route':native_route,'maximum_route_probability':maximum,
        'rejected_by_threshold':rejected,'native_route_probabilities':route_probabilities,
        'native_intent_choice':answers['intent']['choice'],
        'native_intent_probabilities':answers['intent']['probabilities'],
        'native_api_confidence':answers['route'].get('confidence')}
