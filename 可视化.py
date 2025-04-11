import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置全局字体大小和样式
plt.rcParams.update(
    {
        'font.size': 12,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'figure.autolayout': True,
    }
)
sns.set_style("whitegrid")

# 1. 配置路径
data_folder = Path("your_data_folder_path")  # 替换为你的数据文件夹路径
output_folder = Path("output_figures")  # 图表输出文件夹

# 创建输出文件夹（如果不存在）
output_folder.mkdir(parents=True, exist_ok=True)


# 2. 读取Parquet文件
def read_parquet_files(folder_path):
    """从指定文件夹读取所有.parquet文件并合并为一个DataFrame"""
    parquet_files = list(folder_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")

    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        # 检查必要列是否存在
        required_cols = {'age', 'income', 'gender', 'credit_score'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"File {file.name} is missing columns: {missing}")
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # 数据质量检查
    print(f"Loaded data with {len(combined_df):,} rows")
    print("Gender distribution:")
    print(combined_df['gender'].value_counts(normalize=True))

    return combined_df


# 3. 绘图函数
def plot_age_distribution(data, save_path):
    """绘制年龄分布直方图并保存"""
    plt.figure(figsize=(10, 6))
    plt.hist(data['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.8)
    plt.title('Age Distribution', fontsize=14, pad=20)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    output_file = save_path / "age_distribution.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved age distribution plot to: {output_file}")


def plot_credit_vs_income(data, save_path):
    """绘制信用评分vs收入六边形分箱图并保存"""
    plt.figure(figsize=(12, 8))
    hb = plt.hexbin(
        data['credit_score'],
        data['income'],
        gridsize=50,
        cmap='viridis',
        bins='log',
        mincnt=1,  # 避免空箱显示警告
    )
    cb = plt.colorbar(hb, label='Log10(Count)')
    cb.ax.yaxis.set_label_position('left')

    plt.title('Credit Score vs. Income Distribution', fontsize=14, pad=20)
    plt.xlabel('Credit Score', fontsize=12)
    plt.ylabel('Income ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存图片
    output_file = save_path / "credit_vs_income_hexbin.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved credit vs income plot to: {output_file}")


def plot_age_income_quantiles(data, save_path):
    """绘制年龄vs收入分位数图（按性别）并保存"""
    plt.figure(figsize=(12, 8))

    # 准备数据
    data['age_bin'] = pd.cut(data['age'], bins=range(20, 101, 5))

    for gender, color in [('Male', '#1f77b4'), ('Female', '#d62728')]:
        gender_data = data[data['gender'] == gender]

        if len(gender_data) == 0:
            print(f"Warning: No data found for gender '{gender}'")
            continue

        quantiles = (
            gender_data.groupby('age_bin')['income']
            .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            .unstack()
        )
        quantiles = quantiles.reset_index()
        quantiles['age_mid'] = quantiles['age_bin'].apply(lambda x: x.mid)

        # 绘制中位数线
        plt.plot(
            quantiles['age_mid'],
            quantiles[0.5],
            label=f'{gender} Median',
            color=color,
            linewidth=2.5,
            zorder=3,
        )

        # 绘制25%-75%区间
        plt.fill_between(
            quantiles['age_mid'],
            quantiles[0.25],
            quantiles[0.75],
            color=color,
            alpha=0.2,
            label=f'{gender} 25%-75%',
            zorder=2,
        )

        # 绘制5%-95%区间
        plt.fill_between(
            quantiles['age_mid'],
            quantiles[0.05],
            quantiles[0.95],
            color=color,
            alpha=0.1,
            label=f'{gender} 5%-95%',
            zorder=1,
        )

    plt.title('Age vs. Income Distribution by Gender', fontsize=14, pad=20)
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Income ($)', fontsize=12)
    plt.legend(fontsize=10, framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(range(20, 101, 5))
    plt.xlim(20, 95)

    # 保存图片
    output_file = save_path / "age_vs_income_quantiles.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved age vs income quantiles plot to: {output_file}")


# 4. 主执行流程
try:
    # 读取数据
    df = read_parquet_files(data_folder)

    # 生成并保存图表
    plot_age_distribution(df, output_folder)
    plot_credit_vs_income(df, output_folder)
    plot_age_income_quantiles(df, output_folder)

    print("\nAll plots generated successfully!")

except Exception as e:
    print(f"\nError occurred: {str(e)}")
