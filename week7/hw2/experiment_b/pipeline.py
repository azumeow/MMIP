"""依序執行 B 訓練、adapter 轉換、同題測試與比較，逐行保存 UTF-8 日誌。"""
import json
import os
import subprocess
import sys

from classifier.common import save_json
from hw2.experiment_a.prompt import ROOT


def main():
    """只啟動本專案的子程序；任何階段失敗都停止，保留可供檢查的日誌。"""
    venv = ROOT/'.venv-b/Scripts/python.exe'
    if not venv.exists():
        raise FileNotFoundError('請先建立 .venv-b 並安裝 hw2/experiment_b/requirements.txt')
    logs = ROOT/'hw2/results/experiment_b/pipeline'
    logs.mkdir(parents=True,exist_ok=True)
    status = ROOT/'hw2/results/experiment_b/pipeline_status.json'
    adapter = ROOT/'models/experiment_b/adapter.gguf'
    stages = [
        ('train',[str(venv),'-u','-m','hw2.experiment_b.train']),
        ('convert',[str(venv),'-u','-m','hw2.experiment_b.convert_adapter']),
        ('evaluate',[sys.executable,'-u','-m','hw2.experiment_a.local_run','--split','test','--adapter',str(adapter)]),
        ('compare',[str(venv),'-u','-m','hw2.experiment_b.compare']),
    ]
    environment = os.environ.copy();environment['PYTHONIOENCODING']='utf-8'
    for name,command in stages:
        # 已完成訓練時可從轉換／測試繼續，但不覆寫已存在的正式 test 原始回答。
        completed = ROOT/'hw2/results/experiment_b/training/status.json'
        if name=='train' and completed.exists() and json.loads(completed.read_text(encoding='utf-8')).get('completed'):
            print('Reuse completed adapter training.',flush=True)
            continue
        save_json(status,{'phase':name,'completed':False,'command':command})
        print('Starting stage:',name,flush=True)
        with (logs/f'{name}.log').open('w',encoding='utf-8') as stream:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                for line in process.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate();process.wait()
        if code:
            save_json(status,{'phase':'failed','failed_stage':name,'completed':False,'exit_code':code,
                'log':str(logs/f'{name}.log')})
            raise RuntimeError(f'{name} failed; see {logs/name}.log')
    save_json(status,{'phase':'completed','completed':True,'comparison':'hw2/results/experiment_b/test/comparison_ab.md'})
    print('Experiment B completed.',flush=True)


if __name__=='__main__':
    main()
