import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  # 显示所有列
from collections import Counter
import ast  # 解析字符串形式的列表
from sklearn.preprocessing import StandardScaler

def read_df(file_path):
    df = pd.read_csv(file_path)
    return df

def process_artists_train(df):
    # 提取artists列
    artists = df['artists']
    print(artists.dtypes)
    df['artists_list'] = df['artists'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    artists = df['artists_list']
    print(artists.dtypes)
    print(artists.head())

    all_artists = []
    for artist_list in artists:
        all_artists.extend(artist_list)  # 展开所有艺术家到一维列表

    # 统计每个艺术家的出现次数
    artists_counts = Counter(all_artists)

    # 转换为DataFrame并按出现次数降序排序
    result_df = pd.DataFrame(artists_counts.items(), columns=['艺术家', '出现次数'])
    result_df = result_df.sort_values(by='出现次数', ascending=False).reset_index(drop=True)

    print(f"共{len(result_df)}种不同的艺术家")
    print(result_df.head())


    # 统计出现次数的均值、中位数、众数
    count_series = result_df['出现次数']

    mean_count = count_series.mean()  # 均值
    median_count = count_series.median()  # 中位数
    mode_count = count_series.mode()  # 众数（可能有多个，返回Series）

    print("\n出现次数的统计特征")
    print(f"均值：{mean_count:.2f}")
    print(f"中位数：{median_count}")
    if len(mode_count) == 1:
        print(f"众数：{mode_count.iloc[0]}")
    else:
        print(f"众数（多个）：{', '.join(map(str, mode_count.tolist()))}")

    print(count_series.describe())

    """
    根据结果，把在整个数据集中”在artists这一列有出现‘出现次数多于均值数的艺术家’的数据行记为1，否则为0
    这一列命名为”is_occur_mainSinger“
    并统计每一个数据行列表出现歌手的个数，命名为”singer_num“
    """

    # 筛选出出现次数多于均值数的艺术家列表
    main_singers = [artist for artist, count in artists_counts.items() if count > mean_count]
    print(f"\n核心艺术家数量：{len(main_singers)}")

    # 新增列is_occur_mainSinger：判断该行是否包含核心艺术家
    def check_main_singer(artist_list):
        # 只要列表中有一个艺术家在核心列表中，返回1，否则返回0
        return 1 if any(artist in main_singers for artist in artist_list) else 0

    df['is_occur_mainSinger'] = df['artists_list'].apply(check_main_singer)

    # 新增列singer_num：统计每行艺术家列表的歌手个数
    df['singer_num'] = df['artists_list'].apply(len)

    # 验证结果
    print(df[['artists', 'artists_list', 'is_occur_mainSinger', 'singer_num']].head(10))

    # 统计is_occur_mainSinger的分布
    print("\nis_occur_mainSinger列分布：")
    print(df['is_occur_mainSinger'].value_counts())

    # 统计singer_num的分布
    print("\nsinger_num列分布：")
    print(df['singer_num'].value_counts().sort_index())

    df.drop(columns=['artists'], inplace=True)  # inplace=True表示直接在原DataFrame上删除，无需重新赋值
    df.drop(columns=['id'], inplace=True)
    df.drop(columns=['artists_list'], inplace=True)
    print(df.head())

    #df.to_csv('spotify_train_with_new_cols.csv', index=False, encoding='utf-8-sig')

    return df, result_df, main_singers

def process_artists_test(df, train_main_singers):
    """
    用训练集的train_main_singers和核心艺术家列表处理测试集的artists列
    防止数据泄露
    """
    # 解析artists列为列表
    df['artists_list'] = df['artists'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    print(df['artists_list'].head())

    # 新增singer_num列：与训练集逻辑一致
    df['singer_num'] = df['artists_list'].apply(len)
    print("\n测试集singer_num列分布：")
    print(df['singer_num'].value_counts().sort_index())

    # 新增is_occur_mainSinger列：基于训练集核心艺术家判断
    def check_main_singer_test(artist_list):
        return 1 if any(artist in train_main_singers for artist in artist_list) else 0

    df['is_occur_mainSinger'] = df['artists_list'].apply(check_main_singer_test)
    print("\n测试集is_occur_mainSinger列分布：")
    print(df['is_occur_mainSinger'].value_counts())

    df.drop(columns=['artists', 'id', 'artists_list'], inplace=True)
    print(df.head())

    return df


def process_name(df):
    """
    统计每一个数据行歌名的长度，新增一列命名为”name_len“；
    对于每一行数据，查看歌名中是否包含”feat”或“Feat”字样，新增一列命名为“is_featured”，如果有则为1，否则是0；
    对于每一行数据，查看歌名中是否包含”remix”或“Remix”字样，新增一列命名为“is_remixed”，如果有则为1，否则是0
    """

    # 新增name_len列：统计歌名长度（字符数）
    df['name_len'] = df['name'].apply(len)
    print(
        f"歌名长度最大值：{df['name_len'].max()}，最小值：{df['name_len'].min()}，均值：{df['name_len'].mean():.2f}")

    # 新增is_featured列：匹配"feat"或"Feat"
    # # 方式1：严格匹配feat/Feat
    # df['is_featured'] = df['name'].str.contains(r'feat|Feat', regex=True, na=False).astype(int)
    #  方式2：用case=False忽略所有大小写
    df['is_featured'] = df['name'].str.contains('feat', case=False, na=False).astype(int)

    # 新增is_remixed列：匹配"remix"或"Remix"
    # # 方式1
    # df['is_remixed'] = df['name'].str.contains(r'remix|Remix', regex=True, na=False).astype(int)
    # 方式2：忽略所有大小写
    df['is_remixed'] = df['name'].str.contains('remix', case=False, na=False).astype(int)

    # 验证结果
    print(df[['name', 'name_len', 'is_featured', 'is_remixed']].head(10))

    # 统计新增列的分布
    print("\nis_featured列分布：")
    print(df['is_featured'].value_counts().rename(index={0: '不包含', 1: '包含'}))

    print("\nis_remixed列分布：")
    print(df['is_remixed'].value_counts().rename(index={0: '不包含', 1: '包含'}))

    # 统计歌名长度分布
    print("\nname_len分布：")
    print(df['name_len'].value_counts().sort_index().head(10))

    df.drop(columns=['name'], inplace=True)
    print(df.head())

    return df

def process_release_date(df):

    """
    新增一个season列：
    对于release_date这列的每一个数据行，有两种情况：
    1、如果其与year列相同，把season列置为“Unknown”
    2、如果其结构是xx/xx/xx，那么
    如果第一个“/”前的数字xx是{12,1,2}，则把season列置为“winter”；
    如果第一个“/”前的数字xx是{3,4,5}，则把season列置为“spring”；
    如果第一个“/”前的数字xx是{6,7,8}，则把season列置为“summer”；
    如果第一个“/”前的数字xx是{9,10,11}，则把season列置为“autumn”
    """

    df['year'] = df['year'].astype(str)  # 统一转为字符串，避免类型不一致

    # 定义季节判断函数
    def get_season(release_date, year):
        # 情况1：release_date与year相同: Unknown
        if release_date == year:
            return "Unknown"
        # 情况2：判断是否为xx/xx/xx结构
        if "/" in release_date and len(release_date.split("/")) == 3:
            month_str = release_date.split("/")[0]  # 提取第一个/前的数字
            # 处理月份为1位的情况（如1/1/2020，转为01也能正常转int）
            try:
                month = int(month_str)
                if month in [12, 1, 2]:
                    return "winter"
                elif month in [3, 4, 5]:
                    return "spring"
                elif month in [6, 7, 8]:
                    return "summer"
                elif month in [9, 10, 11]:
                    return "autumn"
                else:
                    return "Unknown"  # 月份超出1-12范围
            except ValueError:
                return "Unknown"  # 非数字月份异常值
        # 其他结构（非xx/xx/xx）→ Unknown
        return "Unknown"

    # 新增season列
    df['season'] = df.apply(lambda row: get_season(row['release_date'], row['year']), axis=1)

    # 验证结果
    print(df[['release_date', 'year', 'season']].head(10))

    # 统计season列分布
    print("\nseason列分布：")
    print(df['season'].value_counts())

    # 筛选出season为Unknown的行
    unknown_rows = df[df['season'] == 'Unknown']
    if not unknown_rows.empty:
        # 统计各年份的Unknown数量（按年份排序）
        unknown_year_count = unknown_rows['year'].value_counts().sort_index()
        print("各年份Unknown数量：")
        print(unknown_year_count)

        # 统计各年份Unknown占该年份总数据的比例
        print("\n各年份Unknown占该年份总数据的比例（%）：")
        # 先统计所有数据的年份分布
        total_year_count = df['year'].value_counts()
        # 计算占比
        unknown_year_pct = (unknown_year_count / total_year_count).fillna(0) * 100
        print(unknown_year_pct.round(2))

        print("\nUnknown数量最多的前5个年份：")
        print(unknown_year_count.nlargest(5))

        # 查找Unknown最晚的年份
        # 过滤掉非数字的年份（避免异常值干扰），转为整数型
        valid_unknown_years = unknown_rows['year'].loc[unknown_rows['year'].str.isdigit()]
        if not valid_unknown_years.empty:
            valid_unknown_years_int = valid_unknown_years.astype(int)
            latest_unknown_year = valid_unknown_years_int.max()
            print(f"\nSeason为Unknown的最晚年份：{latest_unknown_year}")
            # 可选：输出该最晚年份的Unknown数量
            latest_year_unknown_count = unknown_year_count[str(latest_unknown_year)]
            print(f"最晚年份{latest_unknown_year}的Unknown数量：{latest_year_unknown_count}")
        else:
            print("\n无有效数字格式的Unknown年份")
    else:
        print("无Season为Unknown的记录")


    df.drop(columns=['release_date'], inplace=True)
    print(df.head())

    return df


def process_year_bin(df):
    """
    按1921-2020音乐发展历程划分Bin，新增era列标注所属阶段
    """
    # # 清洗year列（处理非数字、超出1921-2020范围的异常值）
    # # 转为字符串后过滤数字，再转整数，非数字标记为NaN
    # df['year'] = pd.to_numeric(df['year'], errors='coerce')
    # # 填充异常值：1921以下填充为1921，2020以上填充为2020，NaN按最频年份填充
    # df['year'] = df['year'].clip(lower=1921, upper=2020)
    df['year'] = df['year'].astype(int)  # 转为整数

    # 定义年份区间与对应阶段的映射
    bins = [1920, 1945, 1959, 1979, 1999, 2020]  # 左开右闭
    labels = [
        '1921-1945 早期录音时代（爵士黄金期）',
        '1946-1959 战后复苏与摇滚萌芽期',
        '1960-1979 摇滚黄金期与流派扩张期',
        '1980-1999 电子与嘻哈崛起期',
        '2000-2020 数字时代与全球化期'
    ]

    # 新增music_era列：分箱
    df['era'] = pd.cut(
        df['year'],
        bins=bins,
        labels=labels,
        right=True  # 右闭区间
    )

    # 验证结果
    print("各音乐时代的记录数：")
    era_count = df['era'].value_counts()
    print(era_count)

    # 统计各阶段的年份分布
    print("\n各音乐时代的年份范围验证：")
    for era in labels:
        era_years = df[df['era'] == era]['year'].unique()
        print(f"{era}：年份范围 {min(era_years)} - {max(era_years)}")

    # 简化标签
    df['era'] = pd.cut(
        df['year'],
        bins=bins,
        labels=['EarlyRecord', 'RockBud', 'RockGolden', 'ElectroHipHop', 'DigitalGlobal'],
        right=True
    )

    print(df[['year', 'era']].head(10))
    df.drop(columns=['year'], inplace=True)
    print(df.head())

    return df


def log_transform(df, numeric_cols, epsilon=1e-8):
    """
    对DataFrame中指定的数值特征列进行log变换

    参数:
        df (pd.DataFrame): 输入的数据集
        numeric_cols (list): 需要进行log变换的特征列名列表
        epsilon (float): 用于避免log(0)或log负数的小常数，默认1e-8

    返回:
        pd.DataFrame: 包含log变换后特征的新数据集（不修改原数据）
    """
    # 复制原数据，避免修改输入DataFrame
    transformed_df = df.copy()

    # 校验输入列是否存在
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列不存在于数据集中: {missing_cols}")

    # 校验输入列是否为数值类型
    non_numeric_cols = [col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        raise TypeError(f"以下列不是数值类型，无法进行log变换: {non_numeric_cols}")

    # 对每列进行log变换
    for col in numeric_cols:
        # 处理可能的0或负数（加epsilon确保数值为正）
        transformed_df[col] = np.log(df[col] + epsilon)

    return transformed_df



def mirror_log_transform(df, numeric_cols, epsilon=1e-8, C=None):
    """
    对DataFrame中指定的数值特征列进行镜像log变换（适用于左偏数据）

    变换形式:
        x' = log(C - x + epsilon)

    参数:
        df (pd.DataFrame): 输入的数据集
        numeric_cols (list): 需要进行镜像log变换的特征列名列表
        epsilon (float): 用于避免log(0)的小常数，默认1e-8
        C (float or dict or None):
            - None: 对每一列自动使用该列的最大值
            - float: 所有列使用同一个C
            - dict: 为不同列指定不同的C，例如 {'loudness': 0}

    返回:
        pd.DataFrame: 包含镜像log变换后特征的新数据集（不修改原数据）
    """
    transformed_df = df.copy()

    # 校验输入列是否存在
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下列不存在于数据集中: {missing_cols}")

    # 校验输入列是否为数值类型
    non_numeric_cols = [
        col for col in numeric_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric_cols:
        raise TypeError(f"以下列不是数值类型，无法进行镜像log变换: {non_numeric_cols}")

    for col in numeric_cols:
        # 确定C
        if isinstance(C, dict):
            C_col = C.get(col, df[col].max())
        elif isinstance(C, (int, float)):
            C_col = C
        else:
            C_col = df[col].max()

        # 校验 C - x 是否为正
        if (C_col - df[col] <= 0).any():
            raise ValueError(
                f"列 {col} 中存在值 >= C ({C_col})，"
                "请增大C或检查数据"
            )

        transformed_df[col] = np.log(C_col - df[col] + epsilon)

    return transformed_df

def train_scale_numeric(df):
    """
    把训练集的数值列标准化（Z-score标准化：(x - mean) / std）
    数值列：duration_ms，acousticness，danceability，energy，instrumentalness，liveness，loudness
    speechiness，tempo，valence，key，popularity，singer_num，name_len
    """
    # 定义需要标准化的数值列列表
    numeric_cols = [
        'duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'tempo', 'valence', 'key', 'popularity', 'singer_num', 'name_len'
    ]


    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(df[numeric_cols].head())

    return df, scaler  # 返回标准化后的df和scaler（便于后续测试集复用该scaler）


def encode(df):
    """
    对df中剩下的非数值列进行编码，编码规则：
    1、对于只有二分类的列：mode/explicit/is_occur_mainSinger/is_featured/is_remixed
    保留0/1结构即可
    2、对于多分类列：season/era，进行独热编码
    对于season列，基准定为“Unknown”；对于era列，基准定为“EarlyRecord”
    """
    # 确认二分类列类型为int
    binary_cols = ['mode', 'explicit', 'is_occur_mainSinger', 'is_featured', 'is_remixed']
    for col in binary_cols:
        if df[col].dtype != 'int64':
            df[col] = df[col].astype(int)

    # 多分类列独热编码（指定基准，删除基准列避免共线性）
    season_dummies = pd.get_dummies(df['season'], prefix='season', drop_first=False, dtype=int)
    if 'season_Unknown' in season_dummies.columns:
        season_dummies = season_dummies.drop(columns=['season_Unknown'])

    era_dummies = pd.get_dummies(df['era'], prefix='era', drop_first=False, dtype=int)
    if 'era_EarlyRecord' in era_dummies.columns:
        era_dummies = era_dummies.drop(columns=['era_EarlyRecord'])

    # 合并独热编码列，删除原始多分类列
    df = pd.concat([df, season_dummies, era_dummies], axis=1)
    df.drop(columns=['season', 'era'], inplace=True)

    print(df.head())

    return df


def test_scale_numeric(df, scaler):
    """
    对测试集利用训练集的特征进行标准化
    防止数据泄露
    """
    # 与训练集一致
    numeric_cols = [
        'duration_ms', 'acousticness', 'danceability', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness',
        'tempo', 'valence', 'key', 'popularity', 'singer_num', 'name_len'
    ]

    # 用训练集拟合的scaler标准化
    # 仅transform，不fit，防止数据泄露
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    print(df[numeric_cols].head())

    return df


if __name__=="__main__":
    file_path = "spotify_train.csv"
    df = read_df(file_path)

    df, _, train_main_singers = process_artists_train(df)
    df = process_name(df)
    df = process_release_date(df)
    df = process_year_bin(df)

    # 对长尾分布严重的特征进行log转换
    df = log_transform(df, numeric_cols=['duration_ms', 'singer_num','name_len','instrumentalness'])
    #df = mirror_log_transform(df, numeric_cols=['loudness'], C=df['loudness'].max() + 1e-3)

    df, scaler = train_scale_numeric(df)
    df = encode(df)

    df.to_csv('train_processed.csv', index=False, encoding='utf-8-sig')

    # -----------------------------

    file_path = "spotify_test.csv"
    df = read_df(file_path)

    df = process_artists_test(df, train_main_singers)
    df = process_name(df)
    df = process_release_date(df)
    df = process_year_bin(df)

    # 对偏态分布严重的特征进行log转换
    df = log_transform(df, numeric_cols=['duration_ms', 'singer_num','name_len','instrumentalness'])
    #df = mirror_log_transform(df, numeric_cols=['loudness'], C=df['loudness'].max() + 1e-3)

    df = test_scale_numeric(df, scaler)  # 用训练集scaler标准化
    df = encode(df)

    df.to_csv('test_processed.csv', index=False, encoding='utf-8-sig')
