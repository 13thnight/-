import os
import json
import pandas as pd


INPUT_DIR = "C:/Users/East/Desktop/原数据/30G_data_new"  # 输入目录路径
PROCESSED_DIR = "C:/Users/East/Desktop/预处理数据/30G"  # 输出目录路径
PRODUCT_CATALOG_FILE = "C:/Users/East/Desktop/code/数据挖掘/任务2/product_catalog.json"


def create_category_mapper():
    # 商品分类层级映射
    category_tree = {
        "电子产品": [
            "智能手机",
            "笔记本电脑",
            "平板电脑",
            "智能手表",
            "耳机",
            "音响",
            "相机",
            "摄像机",
            "游戏机",
        ],
        "服装": [
            "上衣",
            "裤子",
            "裙子",
            "内衣",
            "鞋子",
            "帽子",
            "手套",
            "围巾",
            "外套",
        ],
        "食品": [
            "零食",
            "饮料",
            "调味品",
            "米面",
            "水产",
            "肉类",
            "蛋奶",
            "水果",
            "蔬菜",
        ],
        "家居": ["家具", "床上用品", "厨具", "卫浴用品"],
        "办公": ["文具", "办公用品"],
        "运动户外": ["健身器材", "户外装备"],
        "玩具": ["玩具", "模型", "益智玩具"],
        "母婴": ["婴儿用品", "儿童课外读物"],
        "汽车用品": ["车载电子", "汽车装饰"],
    }

    # 反向映射：子类别 -> 父类别
    reverse_mapper = {}
    for parent, children in category_tree.items():
        for child in children:
            reverse_mapper[child] = parent
    return reverse_mapper


def load_product_catalog():
    """加载商品目录并创建映射"""
    with open(PRODUCT_CATALOG_FILE, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    category_mapper = create_category_mapper()

    product_map = {}
    for product in catalog['products']:
        parent_category = category_mapper.get(product['category'], "其他")
        product_map[product['id']] = {
            'parent_category': parent_category,
            'sub_category': product['category'],
            'price': product['price'],
        }
    return product_map


def process_purchase_history(record, product_map):
    """处理单个购买记录"""
    try:
        history = json.loads(record)
        items = history.get('items', [])

        item_details = []
        for item in items:
            product = product_map.get(item['id'], {})
            item_details.append(
                {
                    'parent_category': product.get('parent_category', '未知'),
                    'sub_category': product.get('sub_category', '未知'),
                    'price': product.get('price', 0.0),
                }
            )

        return {
            'payment_method': history.get('payment_method', ''),
            'payment_status': history.get('payment_status', ''),
            'purchase_date': pd.to_datetime(history.get('purchase_date', '')),
            'items_json': json.dumps(item_details, ensure_ascii=False),
            'total_price': sum(item['price'] for item in item_details),
            'item_count': len(items),
        }
    except Exception as e:
        print(f"处理错误：{str(e)}")
        return None


def process_single_file(input_path, output_path, product_map):
    """处理单个文件"""
    df = pd.read_parquet(input_path)

    # 处理每条记录
    processed_data = []
    for _, row in df.iterrows():
        result = process_purchase_history(row['purchase_history'], product_map)
        if result:
            processed_data.append(result)

    # 保存处理结果
    pd.DataFrame(processed_data).to_parquet(output_path)
    print(
        f"已处理完成：{os.path.basename(input_path)} → {os.path.basename(output_path)}"
    )


def main():
    # 创建输出目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 加载商品数据
    product_map = load_product_catalog()
    print(f"已加载 {len(product_map)} 条商品映射信息")

    # 处理所有文件
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".parquet"):
            input_file = os.path.join(INPUT_DIR, filename)
            output_file = os.path.join(PROCESSED_DIR, f"processed_{filename}")
            process_single_file(input_file, output_file, product_map)


if __name__ == "__main__":
    main()
