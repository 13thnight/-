import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置参数
input_dir = "C:/Users/East/Desktop/预处理数据/30G"
output_dir = "C:/Users/East/Desktop/output2/30g/2"
high_value_price = 5000
max_categories = 10  # 最大显示商品类别数
max_payments = 10  # 最大显示支付方式数

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def process_transactions():
    """处理交易数据（优化内存）"""
    category_payments = defaultdict(lambda: defaultdict(int))
    high_value_payments = defaultdict(int)
    payment_types = defaultdict(int)
    category_counts = defaultdict(int)

    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_parquet(file_path, columns=['payment_method', 'items_json'])

            for _, row in df.iterrows():
                try:
                    payment = row['payment_method']
                    items = json.loads(row['items_json'])

                    # 统计支付方式基础频次
                    payment_types[payment] += len(items)

                    for item in items:
                        cat = item['parent_category']
                        # 统计类别支付分布
                        category_payments[cat][payment] += 1
                        category_counts[cat] += 1

                        # 统计高价值支付
                        if item['price'] > high_value_price:
                            high_value_payments[payment] += 1

                except Exception as e:
                    print(f"数据处理错误：{str(e)}")

    return category_payments, high_value_payments, payment_types, category_counts


def visualize_all_distributions(
    category_data, high_value_data, payment_types, category_counts
):
    """整合可视化支付分布"""
    os.makedirs(output_dir, exist_ok=True)

    # 筛选TOP商品类别和支付方式
    top_categories = sorted(category_counts.items(), key=lambda x: -x[1])[
        :max_categories
    ]
    top_payments = sorted(payment_types.items(), key=lambda x: -x[1])[:max_payments]

    # 创建数据矩阵
    payment_names = [p[0] for p in top_payments]
    category_names = [c[0] for c in top_categories]

    # 构建二维数据数组
    data_matrix = []
    for cat in category_names:
        row = [category_data[cat].get(pay[0], 0) for pay in top_payments]
        data_matrix.append(row)

    # 转换为DataFrame
    df = pd.DataFrame(data_matrix, index=category_names, columns=payment_names)

    # 创建堆叠柱状图
    plt.figure(figsize=(18, 10))
    ax = df.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)

    # 图表装饰
    plt.title("各商品大类支付方式分布", fontsize=16)
    plt.xlabel("商品类别", fontsize=12)
    plt.ylabel("交易数量", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='支付方式', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 添加数据标签
    for bars in ax.containers:
        ax.bar_label(
            bars, label_type='center', fmt='%d', padding=2, color='white', fontsize=8
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "1_全品类支付分布.png"), dpi=300, bbox_inches='tight'
    )
    plt.close()

    # 高价值支付分布
    plt.figure(figsize=(12, 8))
    high_value_df = pd.Series(high_value_data)
    high_value_sorted = high_value_df[payment_names].sort_values(ascending=False)

    bars = high_value_sorted.plot(kind='bar', color='#2ca02c')
    plt.title(f"高价值商品支付方式分布（单价>{high_value_price}）", fontsize=14)
    plt.xlabel("支付方式", fontsize=12)
    plt.ylabel("交易数量", fontsize=12)

    # 添加数据标签
    for p in bars.patches:
        height = p.get_height()
        plt.annotate(
            f"{int(height)}",
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center',
            va='bottom',
        )

    plt.savefig(
        os.path.join(output_dir, "2_高价值支付分布.png"), bbox_inches='tight', dpi=300
    )
    plt.close()


def main():
    print("开始处理数据...")
    category_payments, high_value_payments, payment_types, category_counts = (
        process_transactions()
    )

    print("\n生成可视化图表...")
    visualize_all_distributions(
        category_payments, high_value_payments, payment_types, category_counts
    )

    # 保存原始数据
    pd.DataFrame(category_payments).T.to_csv(
        os.path.join(output_dir, "全品类支付分布.csv"), encoding='utf-8-sig'
    )
    pd.Series(high_value_payments).to_csv(
        os.path.join(output_dir, "高价值支付分布.csv"), encoding='utf-8-sig'
    )

    print(f"分析完成！结果保存至：{output_dir}")


if __name__ == "__main__":
    main()
