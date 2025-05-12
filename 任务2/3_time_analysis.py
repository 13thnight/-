import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# 配置参数
input_dir = "C:/Users/East/Desktop/预处理数据/30G"
output_dir = "C:/Users/East/Desktop/output2/30g/3"
top_categories = 10
time_seq_gap = 7

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_time_series_data():
    """加载时间序列数据（优化内存）"""
    date_records = []
    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_parquet(file_path, columns=['purchase_date', 'items_json'])
            for _, row in df.iterrows():
                try:
                    date = pd.to_datetime(row['purchase_date'])
                    items = json.loads(row['items_json'])
                    categories = list(set([item['parent_category'] for item in items]))
                    date_records.append({'date': date, 'categories': categories})
                except Exception as e:
                    print(f"数据处理错误：{str(e)}")
    return pd.DataFrame(date_records)


def analyze_seasonal_patterns(df):
    """分析季节性模式"""
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday + 1
    expanded_df = df.explode('categories')

    quarterly = (
        expanded_df.groupby(['quarter', 'categories']).size().unstack().fillna(0)
    )
    quarterly_top = quarterly[quarterly.sum().nlargest(top_categories).index]

    monthly = expanded_df.groupby(['month', 'categories']).size().unstack().fillna(0)
    monthly_top = monthly[monthly.sum().nlargest(top_categories).index]

    weekday = expanded_df.groupby(['weekday', 'categories']).size().unstack().fillna(0)
    weekday_top = weekday[weekday.sum().nlargest(top_categories).index]

    return quarterly_top, monthly_top, weekday_top


def visualize_seasonal(data_dict):
    """可视化季节性模式"""
    os.makedirs(output_dir, exist_ok=True)

    # 季度趋势
    plt.figure(figsize=(14, 8))
    data_dict['quarterly'].plot(kind='area', alpha=0.8)
    plt.title("季度商品销售趋势TOP{}".format(top_categories))
    plt.xlabel("季度")
    plt.ylabel("销售量")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(
        os.path.join(output_dir, "1_季度趋势.png"), bbox_inches='tight', dpi=300
    )
    plt.close()

    # 月度趋势（修复横坐标）
    plt.figure(figsize=(14, 8))
    ax = data_dict['monthly'].plot(kind='line', marker='o')
    plt.title("月度商品销售趋势TOP{}".format(top_categories))
    plt.xlabel("月份")
    plt.ylabel("销售量")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(
        os.path.join(output_dir, "2_月度趋势.png"), bbox_inches='tight', dpi=300
    )
    plt.close()

    # 周分布
    plt.figure(figsize=(14, 8))
    data_dict['weekday'].plot(kind='bar', stacked=True)
    plt.title("周销售分布TOP{}".format(top_categories))
    plt.xlabel("星期（1-7对应周一到周日）")
    plt.ylabel("销售量")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_dir, "3_周分布.png"), bbox_inches='tight', dpi=300)
    plt.close()


def analyze_sequence_patterns(df):
    """统计先A后B的时序模式"""
    df_sorted = df.sort_values('date')
    transactions = df_sorted['categories'].tolist()
    sequence_counts = defaultdict(int)

    for i in range(len(transactions) - 1):
        for a in transactions[i]:
            for b in transactions[i + 1]:
                sequence_counts[(a, b)] += 1

    seq_list = [
        {'A': a, 'B': b, 'count': cnt} for (a, b), cnt in sequence_counts.items()
    ]
    seq_df = pd.DataFrame(seq_list).sort_values('count', ascending=False).head(30)
    return seq_df


def visualize_sequence_patterns(seq_df, output_dir):
    """可视化时序模式"""
    plt.figure(figsize=(14, 10))
    seq_df['sequence'] = seq_df.apply(lambda x: f"{x['A']} → {x['B']}", axis=1)
    plt.barh(seq_df['sequence'], seq_df['count'], color='skyblue')
    plt.xlabel("出现次数")
    plt.ylabel("时序模式")
    plt.title("Top 30 时序购买模式（A → B）")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "4_时序模式.png"), bbox_inches='tight', dpi=300
    )
    plt.close()


def main():
    print("开始加载数据...")
    time_df = load_time_series_data()
    print(f"已加载 {len(time_df)} 条时间序列记录")

    print("\n分析季节性模式...")
    quarterly, monthly, weekday = analyze_seasonal_patterns(time_df)
    visualize_seasonal({'quarterly': quarterly, 'monthly': monthly, 'weekday': weekday})

    print("\n分析时序购买模式...")
    seq_df = analyze_sequence_patterns(time_df)
    visualize_sequence_patterns(seq_df, output_dir)
    seq_df.to_csv(
        os.path.join(output_dir, "时序模式结果.csv"), index=False, encoding='utf-8-sig'
    )
    print(f"分析结果已保存至：{output_dir}")


if __name__ == "__main__":
    main()
