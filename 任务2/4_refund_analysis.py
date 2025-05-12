import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置参数
input_dir = "C:/Users/East/Desktop/预处理数据/30G"
output_dir = "C:/Users/East/Desktop/output2/30g/4"
target_status = ["已退款", "部分退款"]

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_refund_transactions():
    """加载退款交易数据并提取商品组合"""
    refund_combinations = defaultdict(int)

    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_parquet(file_path, columns=['payment_status', 'items_json'])

            for _, row in df[df['payment_status'].isin(target_status)].iterrows():
                try:
                    items = json.loads(row['items_json'])
                    categories = list(set([item['parent_category'] for item in items]))
                    if len(categories) > 1:  # 只考虑组合情况
                        key = tuple(sorted(categories))
                        refund_combinations[key] += 1
                except Exception as e:
                    print(f"数据处理错误：{str(e)}")

    return refund_combinations


def analyze_refund_combinations(combinations_dict):
    """分析退款组合模式"""
    combinations_list = [
        {'categories': ' & '.join(k), 'count': v} for k, v in combinations_dict.items()
    ]
    df = pd.DataFrame(combinations_list)
    return df.sort_values('count', ascending=False).head(30)


def visualize_refund_patterns(df):
    """可视化退款组合模式"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(16, 12))
    ax = df.plot.barh(x='categories', y='count', color='#1f77b4', legend=False)

    # 添加数据标签
    for i, (value, name) in enumerate(zip(df['count'], df['categories'])):
        ax.text(value + 0.5, i, str(value), ha='left', va='center', fontsize=10)

    # 图表装饰
    plt.title("高频退款商品组合TOP30", fontsize=14)
    plt.xlabel("出现次数", fontsize=12)
    plt.ylabel("商品组合", fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 保存结果
    img_path = os.path.join(output_dir, "refund_combinations.png")
    csv_path = os.path.join(output_dir, "refund_combinations.csv")
    plt.savefig(img_path, bbox_inches='tight', dpi=300)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至：\n{img_path}\n{csv_path}")


def main():
    print("开始加载退款数据...")
    combinations = load_refund_transactions()
    print(f"发现有效退款组合：{len(combinations)}种")

    if not combinations:
        print("没有找到退款组合记录")
        return

    print("\n分析高频退款组合...")
    top_combinations = analyze_refund_combinations(combinations)
    visualize_refund_patterns(top_combinations)


if __name__ == "__main__":
    main()
