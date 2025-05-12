import os
import json
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# 配置参数
input_dir = "C:/Users/East/Desktop/预处理数据/30G"
output_dir = "C:/Users/East/Desktop/output2/30g/1"
target_category = "电子产品"  # 目标分析类别
max_combo_length = 3  # 分析的最大组合长度
top_n = 50  # 可视化显示前N个组合

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_and_count_combos():
    """加载数据并统计组合频率"""
    combo_counter = defaultdict(int)

    for file in os.listdir(input_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_parquet(file_path, columns=['items_json'])

            for items_json in df['items_json']:
                try:
                    items = json.loads(items_json)
                    categories = list(set([item['parent_category'] for item in items]))

                    # 生成所有可能组合（长度2到max_combo_length）
                    for r in range(2, max_combo_length + 1):
                        for combo in combinations(categories, r):
                            sorted_combo = tuple(sorted(combo))  # 标准化排序
                            combo_counter[sorted_combo] += 1

                except Exception as e:
                    continue

    return combo_counter


def format_combo_name(combo):
    """格式化组合名称，确保电子产品在首位"""
    categories = list(combo)
    # 如果包含电子产品则调整顺序
    if '电子产品' in categories:
        # 将电子产品移到首位
        categories.remove('电子产品')
        sorted_rest = sorted(categories)  # 其他类别按字母排序
        return '电子产品 & ' + ' & '.join(sorted_rest)
    else:
        # 普通组合按字母排序
        return ' & '.join(sorted(categories))


def filter_electronics_combos(combo_counter):
    """筛选并格式化含电子产品的组合"""
    electronics_combos = {}
    for combo, count in combo_counter.items():
        if '电子产品' in combo:
            # 使用标准化格式命名
            combo_str = format_combo_name(combo)
            electronics_combos[combo_str] = count
    return electronics_combos


def visualize_combos(combos_dict, title, filename):
    """可视化组合频率"""
    os.makedirs(output_dir, exist_ok=True)

    # 转换为DataFrame并排序
    df = (
        pd.DataFrame(
            {'组合': list(combos_dict.keys()), '出现次数': list(combos_dict.values())}
        )
        .sort_values('出现次数', ascending=False)
        .head(top_n)
    )

    # 可视化
    plt.figure(figsize=(14, 10))
    bars = plt.barh(df['组合'], df['出现次数'], color='#2ca02c')
    plt.title(f"{title}\n(TOP {top_n} 高频组合)")
    plt.xlabel("出现次数")
    plt.gca().invert_yaxis()  # 降序排列

    # 添加数据标签
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(df['出现次数']) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{width:,}',
            va='center',
        )

    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    print("开始统计组合频率...")
    combo_counter = load_and_count_combos()
    print(f"发现 {len(combo_counter)} 种不同组合")

    if not combo_counter:
        print("无有效组合数据，分析终止")
        return

    print("\n可视化全部高频组合...")
    all_combos = {f"{' & '.join(k)}": v for k, v in combo_counter.items()}
    visualize_combos(all_combos, "全部商品组合", "1_all_combos.png")

    print("\n筛选电子产品相关组合...")
    electronics_combos = filter_electronics_combos(combo_counter)
    print(f"找到 {len(electronics_combos)} 个相关组合")

    if electronics_combos:
        visualize_combos(
            electronics_combos, "含电子产品的组合", "2_electronics_combos.png"
        )
    else:
        print("无电子产品相关组合")

    # 保存CSV
    pd.DataFrame(
        {'组合': list(all_combos.keys()), '出现次数': list(all_combos.values())}
    ).to_csv(
        os.path.join(output_dir, "all_combos.csv"), index=False, encoding='utf-8-sig'
    )

    print(f"分析完成！结果保存至: {output_dir}")


if __name__ == "__main__":
    main()
