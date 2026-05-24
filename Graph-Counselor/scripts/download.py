# download.py
from modelscope import snapshot_download
import os

# 1. 设置保存路径
save_dir = '/root/Graph-Counselor/Graph-Counselor/models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# print("🚀 开始下载 Qwen2.5-7B-Instruct...")

# # 2. 开始下载
# try:
#     model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir=save_dir)
#     print("\n✅ 下载成功！")
#     print(f"📂 请务必复制这个路径到你的 sh 脚本里: {model_dir}")
# except Exception as e:
#     print(f"❌ 下载失败: {e}")
try:
    # ModelScope 上的 ID 可能略有不同，通常是这个
    model_dir = snapshot_download('AI-ModelScope/all-mpnet-base-v2', cache_dir=save_dir)
    print(f"✅ 下载成功！")
    print(f"📂 模型绝对路径: {model_dir}")
except Exception as e:
    print(f"❌ 下载失败: {e}")