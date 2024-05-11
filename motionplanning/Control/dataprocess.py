import pandas as pd
import os

def read_last_line_of_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return lines[-1].strip().split()

def save_to_excel(folder_path, output_excel):
    df = pd.DataFrame(columns=['参数设置', 'err_d', 'err_theta', 'err_dist'])

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            try:
                last_line_data = read_last_line_of_txt(file_path)
                if len(last_line_data) == 3:
                    # 获取文件名（不带扩展名）
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    # 将文件名和其他数据添加到DataFrame中
                    df = df.append(pd.Series([file_name_without_extension] + last_line_data, index=df.columns), ignore_index=True)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    df.to_excel(output_excel, index=False)

folder_path = 'output/eval/exp2'  # 替换为你的文件夹路径
output_excel = 'output/eval/exp2.xlsx'  # 你希望保存的Excel文件名

save_to_excel(folder_path, output_excel)
