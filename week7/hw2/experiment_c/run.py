"""C 的入口：沿用 A 的模型與評分器，只啟用生成階段的 Schema 約束。"""
import subprocess
import sys

from hw2.experiment_a.prompt import ROOT


def main():
    """不載入 LoRA、不重新訓練、不修復 JSON；結果與 A/B 分別保存。"""
    subprocess.run([sys.executable,"-u","-m","hw2.experiment_a.local_run","--split","test",
        "--constrained-decoding"],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-m","hw2.experiment_c.compare"],cwd=ROOT,check=True)


if __name__=="__main__":
    main()
