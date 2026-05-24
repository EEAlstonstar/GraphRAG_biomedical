import json
import re
import os
import sys

# 尝试导入 graph_fewshots
try:
    from graph_fewshots import PLAN_EXAMPLES, PLAN_ONLY_EXAMPLES
except ImportError:
    print("❌ 错误：无法导入 graph_fewshots.py。请确保此脚本与 graph_fewshots.py 在同一目录下。")
    sys.exit(1)

def convert_to_json_smart(dataset_key='pubmed', output_filename='few_shot_bank.json'):
    # 路径设置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = 'biomedical' # 或者是 'pubmed'，根据你的文件夹名称
    output_dir = os.path.join(current_dir, "../../data/processed_data", target_folder)
    output_path = os.path.join(output_dir, output_filename)

    print(f"🔍 正在处理数据集: {dataset_key}")

    # 获取原始文本
    if dataset_key in PLAN_EXAMPLES:
        raw_text = PLAN_EXAMPLES[dataset_key]
    elif dataset_key in PLAN_ONLY_EXAMPLES:
        raw_text = PLAN_ONLY_EXAMPLES[dataset_key]
    else:
        print(f"❌ 未找到 Key: {dataset_key}")
        return

    bank = []

    # === 核心修改逻辑 ===
    # 不使用 (END OF EXAMPLE) 分割，而是使用 "Definition of the graph:" 分割
    # 这样可以把粘在一起的例子分开
    
    # 1. 先用 split 切开
    split_pattern = "Definition of the graph:"
    parts = raw_text.split(split_pattern)

    for part in parts:
        clean_part = part.strip()
        # 跳过空字符串或过短的碎片
        if len(clean_part) < 50:
            continue
        
        # 2. 因为 split 会把分隔符切掉，我们需要把它加回去
        # 注意：最后一个例子可能带有 (END OF EXAMPLE)，我们需要保留或清理它
        full_trace = split_pattern + "\n" + clean_part
        
        # 3. 清理末尾多余的 (END OF EXAMPLE) 标记，保持干净，稍后统一加
        full_trace = full_trace.replace("(END OF EXAMPLE)", "").strip()
        
        # 4. 加上统一的结尾符（为了格式统一）
        full_trace += "\n(END OF EXAMPLE)"

        # 5. 提取 Question
        match = re.search(r'Question:\s*(.+?)(\n|$)', full_trace)
        if match:
            question_text = match.group(1).strip()
            bank.append({
                "question": question_text,
                "trace": full_trace
            })
            print(f"  -> 成功提取示例: {question_text[:40]}...")
        else:
            print("  ⚠️ 跳过一段无法提取 Question 的文本")

    # === 保存 ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bank, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 修正完成！共生成 {len(bank)} 个独立示例。")
    print(f"📁 保存路径: {output_path}")
    print("🔴 重要提示：虽然现在有 3 个独立例子了，但为了展示'动态检索'的效果，强烈建议你在生成的 JSON 里复制这几项，修改其中的 Question 和 Answer，凑够至少 5-10 个例子。")

if __name__ == "__main__":
    # 请确保这里的 key 和你 fewshots 文件里的一致
    DATASET_KEY = 'biomedical' 
    convert_to_json_smart(dataset_key=DATASET_KEY)