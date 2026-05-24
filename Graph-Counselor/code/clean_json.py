import json
import os

# 确保路径对准你的文件
file_path = "../../data/processed_data/biomedical/few_shot_bank.json"

def force_clean():
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    print(f"正在读取文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0
    for i, item in enumerate(data):
        trace = item.get('trace', '')
        
        # 核心逻辑：不管前面是什么，只要有 "Question:"，就只保留从它开始的部分
        # 注意：这里匹配 "Question:" (带冒号)
        idx = trace.find("Question:")
        
        if idx != -1:
            # 如果 idx > 0 说明前面有废话（Definition...）
            if idx > 0:
                # 截取：只保留 Question: 及之后的内容
                clean_trace = trace[idx:]
                item['trace'] = clean_trace
                cleaned_count += 1
                if i == 0:
                    print(f"示例预览（清洗后）: {clean_trace[:50]}...")
        else:
            print(f"⚠️ 第 {i} 个示例没有找到 'Question:'，跳过。")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 成功清洗了 {cleaned_count} 个示例！")
    print("现在 JSON 里应该没有 'Definition of the graph' 了。")

if __name__ == "__main__":
    force_clean()