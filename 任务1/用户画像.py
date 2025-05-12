import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
import matplotlib
import time
import gc

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12


def extract_purchase_amount(purchase_history):
    """从purchase_history中提取总消费金额"""
    try:
        ph = json.loads(purchase_history)
        return ph.get('average_price', 0) * len(ph.get('items', []))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return np.nan


def read_and_process_data(folder_path):
    """读取数据并直接处理为所需的两列"""
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not files:
        raise ValueError(f"No parquet files found in {folder_path}")

    # 初始化空DataFrame用于存储结果
    final_df = pd.DataFrame(columns=['income', 'total_purchase_amount'])

    for file in files:
        file_path = os.path.join(folder_path, file)
        # 只读取需要的列
        df = pd.read_parquet(file_path, columns=['income', 'purchase_history'])

        # 直接处理为所需的两列数据
        df['total_purchase_amount'] = df['purchase_history'].apply(
            extract_purchase_amount
        )
        final_df = pd.concat(
            [final_df, df[['income', 'total_purchase_amount']]], ignore_index=True
        )

        # 及时释放内存
        del df
        gc.collect()

    return final_df


def plot_scatter(df, output_folder):
    """绘制收入与总消费金额的散点图"""
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='income', y='total_purchase_amount', data=df, alpha=0.6, s=15)
    plt.title('收入与消费金额关系', pad=20)
    plt.xlabel('收入')
    plt.ylabel('总消费金额')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'income_vs_purchase.png'), dpi=300)
    plt.close()


def perform_clustering(df, output_folder):
    """进行聚类分析并可视化"""
    os.makedirs(output_folder, exist_ok=True)

    # 删除缺失值
    cluster_df = df.dropna().copy()
    if len(cluster_df) < 2:
        return

    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_df)

    # K-means聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    cluster_df['cluster'] = clusters

    # 定义聚类标签
    cluster_labels = {
        0: '低收入低消费',
        1: '高收入高消费',
        2: '低收入高消费',
        3: '高收入低消费',
    }
    cluster_df['cluster_label'] = cluster_df['cluster'].map(cluster_labels)

    # 绘制分组散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='income',
        y='total_purchase_amount',
        hue='cluster_label',
        data=cluster_df,
        palette='viridis',
        alpha=0.7,
        s=15,
    )
    plt.title('收入与消费金额关系(聚类分组)', pad=20)
    plt.xlabel('收入')
    plt.ylabel('总消费金额')
    plt.legend(title='用户群体')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'clustered_scatter.png'), dpi=300)
    plt.close()

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    cluster_counts = cluster_df['cluster_label'].value_counts()
    plt.pie(
        cluster_counts,
        labels=cluster_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'black'},
    )
    plt.title('用户群体分布', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'user_segments.png'), dpi=300)
    plt.close()


def main(input_folder, output_folder):
    """主函数"""
    start_time = time.time()

    try:
        print("正在读取并处理数据...")
        df = read_and_process_data(input_folder)

        print("正在绘制基础散点图...")
        plot_scatter(df, output_folder)

        print("正在进行聚类分析...")
        perform_clustering(df, output_folder)

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
    finally:
        if 'df' in locals():
            del df
        gc.collect()

        elapsed_time = (time.time() - start_time) * 3
        print(f"可视化结果已保存到: {output_folder}")
        print(f"总运行时间: {elapsed_time:.2f}秒")


if __name__ == "__main__":
    input_folder = "C:/Users/East/Desktop/数据挖掘/30G"
    output_folder = "C:/Users/East/Desktop/数据挖掘/outputs/30G"

    main(input_folder, output_folder)
