from pathlib import Path
import subprocess
import sys

# main.py 放在 week1 資料夾。
week1 = Path(__file__).resolve().parent
name = input("請輸入要執行的題目（例如 quiz_1）：").strip()

if not (name.startswith("quiz_") and name[5:].isdigit()):
    print("請輸入 quiz_1、quiz_2 等題目名稱。")
    sys.exit(1)

notebook = week1 / "Code" / f"{name}.ipynb"

if not notebook.is_file():
    print(f"找不到檔案：{notebook}")
    sys.exit(1)

# 依序執行所有 cell，將結果存回同一個 notebook。
result = subprocess.run(
    [sys.executable, "-m", "nbconvert", "--to", "notebook",
     "--execute", "--inplace", "--ExecutePreprocessor.timeout=-1",
     str(notebook)],
    cwd=notebook.parent,
)

if result.returncode == 0:
    print(f"{name} 執行完成！請開啟 notebook 查看結果。")
else:
    print("執行失敗，請查看上方錯誤訊息。")

sys.exit(result.returncode)
