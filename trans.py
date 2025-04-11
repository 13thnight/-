import pandas as pd
import pyarrow.parquet as pq
import os
import csv
import time
from tqdm import tqdm  # 进度条支持（可选）


def convert_parquet_to_csv(input_parquet, output_csv, sample_rows=10000):
    """
    高效读取大型Parquet文件的前N行并保存为CSV（解决乱码问题）

    参数:
        input_parquet: Parquet文件路径（如 "C:/path/to/file.parquet"）
        output_csv: 输出CSV路径
        sample_rows: 提取行数（默认10000）
    """
    print(f"开始处理文件: {input_parquet}")
    start_time = time.time()

    try:
        # 方法1：使用PyArrow直接读取（内存效率更高）
        parquet_file = pq.ParquetFile(input_parquet)

        # 获取总行数信息
        total_rows = parquet_file.metadata.num_rows
        print(f"文件总行数: {total_rows:,} | 目标提取行数: {sample_rows}")

        # 分批次读取（避免内存不足）
        batches = []
        remaining_rows = sample_rows

        for batch in tqdm(
            parquet_file.iter_batches(batch_size=100000), desc="读取进度"
        ):
            batches.append(batch)
            remaining_rows -= len(batch)
            if remaining_rows <= 0:
                break

        # 合并批次并截取所需行数
        table = pa.Table.from_batches(batches)
        df = table.to_pandas().head(sample_rows)

    except Exception as e:
        print(f"PyArrow读取失败: {e} \n回退到Pandas方式...")
        # 方法2：使用Pandas作为备选
        df = pd.read_parquet(input_parquet).head(sample_rows)

    # 处理可能的乱码问题
    for col in df.columns:
        # 处理二进制列
        if df[col].dtype == object and isinstance(df[col].iloc[0], (bytes, bytearray)):
            df[col] = df[col].apply(
                lambda x: x.decode('utf-8', errors='replace') if x else None
            )
        # 处理其他非字符串列
        elif not pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)

    # 保存CSV（解决中文乱码）
    df.to_csv(
        output_csv,
        index=False,
        encoding='utf-8-sig',  # 添加BOM头（兼容Excel）
        quoting=csv.QUOTE_NONNUMERIC,  # 非数字内容加引号
        escapechar='\\',  # 转义特殊字符
    )

    # 打印统计信息
    end_time = time.time()
    input_size = os.path.getsize(input_parquet) / (1024**3)  # GB
    output_size = os.path.getsize(output_csv) / (1024**2)  # MB

    print(f"\n转换完成！")
    print(f"- 输入文件: {input_parquet} ({input_size:.2f} GB)")
    print(f"- 输出文件: {output_csv} ({output_size:.2f} MB)")
    print(f"- 实际提取行数: {len(df)}")
    print(f"- 耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 配置路径（根据实际修改）
    input_parquet = "C:/Users/East/Desktop/30G_data/part-00000.parquet"
    output_csv = "C:/Users/East/Desktop/30G_data/part-00000.csv"

    # 执行转换
    convert_parquet_to_csv(input_parquet, output_csv)
