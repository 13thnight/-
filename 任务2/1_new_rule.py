import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 配置参数
input_dir = "C:/Users/East/Desktop/test2"
output_dir = "C:/Users/East/Desktop/test2/output"
target_category = "电子产品"
max_combo_length = 3
top_n = 50
sample_ratio = 0.1  # 抽样比例
min_support = 0.002  # 最小支持度阈值
min_confidence = 0.05  # 最小置信度阈值

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_and_sample_data():
    """抽样加载数据"""
    all_transactions = []

    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            # 抽样读取数据
            df = pd.read_parquet(file_path, columns=['items_json']).sample(
                frac=sample_ratio
            )

            for items_json in df['items_json']:
                try:
                    items = json.loads(items_json)
                    categories = list(set([item['parent_category'] for item in items]))
                    if len(categories) >= 2:  # 只保留有组合的订单
                        all_transactions.append(categories)
                except:
                    continue
    return all_transactions


def analyze_association_rules(transactions):
    """使用Apriori算法分析关联规则"""
    # 转换事务数据格式
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    # 生成关联规则
    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )

    # 格式化规则输出
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

    return rules


def filter_electronics_rules(rules):
    """筛选含电子产品的有效规则"""
    electronics_rules = rules[
        (rules['antecedents'].str.contains(target_category))
        | (rules['consequents'].str.contains(target_category))
    ]

    # 计算规则重要度指标
    electronics_rules = electronics_rules[
        ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    ].sort_values('support', ascending=False)

    return electronics_rules


def visualize_rules(rules, filename):
    """可视化关联规则"""
    plt.figure(figsize=(14, 10))
    rules['rule'] = rules.apply(
        lambda x: f"{x['antecedents']} → {x['consequents']}", axis=1
    )

    # 取TOP20规则可视化
    top_rules = rules.head(20)
    plt.barh(top_rules['rule'], top_rules['support'], color='#1f77b4')
    plt.title(f"Top 20 关联规则 (支持度≥{min_support}, 置信度≥{min_confidence})")
    plt.xlabel("支持度")
    plt.gca().invert_yaxis()

    # 添加数据标签
    for i, v in enumerate(top_rules['support']):
        plt.text(v + 0.005, i, f"{v:.2%}", color='black', va='center')

    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # 数据加载与抽样
    print("开始抽样加载数据...")
    transactions = load_and_sample_data()
    print(f"抽样后有效订单数：{len(transactions):,}")

    # 关联规则分析
    print("\n分析关联规则...")
    rules = analyze_association_rules(transactions)
    electronics_rules = filter_electronics_rules(rules)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    electronics_rules.to_csv(
        os.path.join(output_dir, "electronics_rules.csv"),
        index=False,
        encoding='utf-8-sig',
    )

    # 可视化
    print("\n可视化关联规则...")
    visualize_rules(electronics_rules, "electronics_association_rules.png")

    print(f"分析完成！结果保存至：{output_dir}")


if __name__ == "__main__":
    main()
