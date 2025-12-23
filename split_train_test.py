import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "spotify-data.csv"
df = pd.read_csv(file_path)

def missing_values(df):
    """缺失值统计"""
    miss_cnt = df.isnull().sum()
    miss_ratio = (df.isnull().mean() * 100).round(2)

    miss_stats = pd.DataFrame({
        '缺失值数量': miss_cnt,
        '缺失值占比': miss_ratio
    }).sort_values('缺失值数量', ascending=False)

    total_cells = df.shape[0] * df.shape[1]
    total_miss = miss_cnt.sum()
    print(f"总单元格数：{total_cells}")
    print(f"总缺失值数：{total_miss}")

    return miss_stats

miss = missing_values(df)

### 无缺失值

def drop_dup(df):
    print(f"去除重复值前：{len(df)}")
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)# inplace=True直接修改原df
    print(f"去除重复值后：{len(df)}")

drop_dup(df)

### 无重复值

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)

print(f"原始数据集总条数：{len(df)}")
print(f"训练集条数：{len(train_df)}（占比 {len(train_df)/len(df):.2%}）")
print(f"测试集条数：{len(test_df)}（占比 {len(test_df)/len(df):.2%}）")

train_df.to_csv("spotify_train.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("spotify_test.csv", index=False, encoding="utf-8-sig")