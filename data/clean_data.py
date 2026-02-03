import pandas as pd
import numpy as np
import ast
import re

def clean_and_save(input_path, output_path):
    print(f"读取数据: {input_path}")
    # 使用 | 作为分隔符，如果某些行有问题，error_bad_lines=False (旧版pandas) 或 on_bad_lines='skip' (新版pandas)
    try:
        df = pd.read_csv(input_path, sep=',', on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(input_path, sep=',', error_bad_lines=False)

    records = []
    error_count = 0
    nan_werks_count = 0

    print("开始清洗与转换...")
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"处理进度: {idx}/{total_rows}")

        # 1. 处理 werks 和 material_code (构建 unique_id)
        # print(row)
        werks = row.get('werks')
        material = row.get('material_code')
        # print(werks, material)
        # 检查 NaN
        if pd.isna(werks) or pd.isna(material):
            nan_werks_count += 1
            continue
        
        
        try:
            # 视为字符串处理
            # 即使是 '1301.0'，我们也可以选择保留或者去掉 .0
            # 这里为了 ID 干净，去掉可能存在的 '.0' 后缀
            werks_str = str(werks).strip()
            if werks_str.endswith('.0'):
                werks_str = werks_str[:-2]
                
            material_str = str(material).strip()
            uid = f"{werks_str}_{material_str}"
        except Exception:
            # 如果无法转换，跳过
            nan_werks_count += 1
            continue

        # 2. 处理 history_demand
        demand_str = row.get('history_demand')
        
        if pd.isna(demand_str) or demand_str == '':
            continue
        
        # 预处理字符串
        # 替换 np.float64(...) 为纯数字
        clean_str = re.sub(r'np\.float64\((.*?)\)', r'\1', str(demand_str))
        # 有时候可能出现 nan (非字符串形式)，跳过
        if 'nan' == clean_str.lower():
            continue

        try:
            # 尝试解析字典
            demand_dict = ast.literal_eval(clean_str)
            
            if not isinstance(demand_dict, dict):
                continue
                
            for date_str, val in demand_dict.items():
                # 转换日期格式 '202303' -> '2023-03-01'
                try:
                    dt = pd.to_datetime(date_str, format='%Y%m')
                    records.append({
                        'unique_id': uid,
                        'ds': dt,
                        'y': float(val)
                    })
                except:
                    # 日期格式不对，跳过该条记录
                    pass
                    
        except Exception as e:
            # 解析失败 (例如格式错乱)
            error_count += 1
            continue

    print(f"\n清洗统计:")
    print(f"- 原始行数: {total_rows}")
    print(f"- 跳过 werks/material 为空的行: {nan_werks_count}")
    print(f"- 解析 history_demand 失败的行: {error_count}")
    
    if not records:
        print("错误: 没有解析出任何有效数据！")
        return

    # 转换为 DataFrame
    long_df = pd.DataFrame(records)
    
    # 3. 再次检查数据有效性
    long_df = long_df.dropna()
    long_df = long_df.sort_values(['unique_id', 'ds'])
    
    print(f"- 解析后有效记录数 (Long Format): {len(long_df)}")
    print(f"- 包含 Unique IDs: {long_df['unique_id'].nunique()}")

    # 4. 保存
    print(f"正在保存清洗后的数据至: {output_path}")
    long_df.to_csv(output_path, index=False)
    print("完成!")

if __name__ == "__main__":
    input_file = './data_all.csv'
    output_file = './data_cleaned.csv'
    clean_and_save(input_file, output_file)