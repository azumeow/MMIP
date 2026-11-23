"""使用同版本 llama.cpp 的官方轉換程式，將 PEFT LoRA 轉為 GGUF adapter。"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from classifier.common import save_json
from hw2.experiment_a.prompt import ROOT


def setup_converter():
    """只解壓 Python 轉換程式與 gguf 套件，不執行下載的 shell 安裝腳本。"""
    destination = ROOT / "tools/llama-converter-b12359"
    if (destination / "convert_lora_to_gguf.py").exists():
        return destination
    url = "https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/b12359"
    session = requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=.5)))
    response = session.get(url,timeout=(8,60)); response.raise_for_status()
    payload = response.content
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        for member in zipped.infolist():
            relative = member.filename.split('/',1)[-1]
            selected = relative == 'convert_lora_to_gguf.py' or relative.startswith(('conversion/','gguf-py/gguf/'))
            if not selected or member.is_dir():
                continue
            target = (destination/relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Converter archive path outside destination')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(zipped.read(member))
            extracted.append(relative)
    save_json(destination/'source_manifest.json',{'url':url,'tag':'b12359',
        'archive_sha256':hashlib.sha256(payload).hexdigest(),'files':extracted})
    return destination


def prepare_conversion_base():
    """LoRA 轉換只需原始架構；另存 metadata，避免工具嘗試反量化整份 NF4 模型。"""
    source = ROOT/'models/qwen3-8b-nf4'
    destination = ROOT/'models/experiment_b/conversion_base'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json',
                 'generation_config.json','merges.txt','vocab.json','chat_template.jinja']:
        if (source/name).exists():
            shutil.copyfile(source/name,destination/name)
    import json
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    config.pop('quantization_config',None)
    save_json(destination/'config.json',config)
    # 此目錄沒有基礎模型權重，不宣稱已把 NF4 轉回全精度；只是提供 adapter 的維度與命名規則。
    save_json(destination/'conversion_note.json',{'source':str(source),
        'purpose':'Architecture/tokenizer metadata only for LoRA conversion; base weights are not converted.',
        'quantization_config_omitted':True})
    return destination


def main():
    """轉換只使用 adapter 權重與基礎模型 config，不合併或改寫 A 的 GGUF。"""
    converter = setup_converter()
    adapter = ROOT / 'models/experiment_b/adapter'
    if not (adapter/'adapter_model.safetensors').exists():
        raise FileNotFoundError('LoRA adapter 尚未訓練完成')
    output = ROOT/'models/experiment_b/adapter.gguf'
    base_config = prepare_conversion_base()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(converter/'gguf-py')+os.pathsep+environment.get('PYTHONPATH','')
    subprocess.run([sys.executable,str(converter/'convert_lora_to_gguf.py'),
        '--base',str(base_config),'--outfile',str(output),
        '--outtype','f16',str(adapter)],check=True,env=environment,cwd=converter)
    save_json(output.with_suffix('.source.json'),{'adapter_path':str(adapter),
        'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'base_inference_model':'Qwen3-8B-Q4_K_M.gguf (same as A)',
        'converter':'llama.cpp b12359'})
    print('Converted LoRA:',output)


if __name__ == '__main__':
    main()
