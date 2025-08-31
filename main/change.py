import nbformat as nbf
import os

def py_to_ipynb(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()

    # 创建一个新的 Notebook
    nb = nbf.v4.new_notebook()

    # 将代码分成单独的代码单元格
    cells = []
    for cell in code.split('\n\n'):
        cells.append(nbf.v4.new_code_cell(cell))

    nb['cells'] = cells

    # 保存为 .ipynb 文件
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def convert_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                py_file = os.path.join(root, file)
                ipynb_file = os.path.splitext(py_file)[0] + '.ipynb'
                print(f"Converting {py_file} to {ipynb_file}")
                py_to_ipynb(py_file, ipynb_file)

# 使用示例
directory = r"E:\Project\17 轴承跨个体\03 code\share"  # 替换为你的目录路径
convert_directory(directory)
