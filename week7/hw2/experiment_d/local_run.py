"""在本機啟動 Clef 原生決策服務，固定問句評估後關閉自己建立的服務。"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from classifier.common import ROOT,save_json
from .setup_model import NAME


def main():
    """--fit 自動安排顯存；原生 Clef 的整個 prompt 必須放入單一 ubatch。"""
    output=ROOT/'hw2/results/experiment_d/test'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'raw_results.jsonl').exists():raise FileExistsError('已有正式 D 結果，不覆寫')
    model=ROOT/'models/clef-flash-gguf'/NAME
    exe=next((ROOT/'tools/llama-b12359-cuda').rglob('llama-server.exe'))
    if not model.exists():raise FileNotFoundError('請先執行 python -m hw2.experiment_d.setup_model')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8082))==0:raise RuntimeError('8082 已被其他程序使用')
    command=[str(exe),'-m',str(model),'--alias','clef-flash-q4_k_m','--host','127.0.0.1','--port','8082',
        '-c','4096','--parallel','1','--batch-size','2048','--ubatch-size','2048',
        '--fit','on','--fit-target','512','--no-mmproj','--jinja']
    save_json(output/'server_config.json',{'command':command,'host':'127.0.0.1','port':8082,
        'gpu_policy':'Automatic fit; exact layer offload counts may be absent at default log verbosity',
        'ubatch_policy':'Entire native decision prompt must fit in one 2048-token ubatch'})
    shutil.copyfile(model.parent/'source_manifest.json',output/'model_source.json')
    environment=os.environ.copy()
    environment['PATH']='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/bin'+os.pathsep+environment.get('PATH','')
    with (output/'server.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=environment,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+240
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Clef 啟動失敗，請看 server.log')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8082/health',timeout=2) as response:
                        if response.status==200:break
                except (urllib.error.URLError,TimeoutError):time.sleep(1)
            else:raise TimeoutError('Clef 模型載入逾時')
            with urllib.request.urlopen('http://127.0.0.1:8082/v1/models',timeout=5) as response:
                capabilities=json.load(response)
            save_json(output/'server_capabilities.json',capabilities)
            if 'decisions' not in json.dumps(capabilities):
                raise RuntimeError('模型沒有宣告原生 decisions 能力，不能冒稱 System One')
            # 留存實際服務設定與 build 資訊；不把設定內容傳入分類問題。
            with urllib.request.urlopen('http://127.0.0.1:8082/props',timeout=5) as response:
                save_json(output/'server_properties.json',json.load(response))
            print('Native Clef-Flash ready.',flush=True)
            subprocess.run([sys.executable,'-u','-m','hw2.experiment_d.run'],cwd=ROOT,check=True)
        finally:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    subprocess.run([sys.executable,'-m','hw2.experiment_d.compare'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
