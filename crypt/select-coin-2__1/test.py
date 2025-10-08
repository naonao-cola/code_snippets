import pandas as pd

from core.utils.path_kit import get_file_path

if __name__ == '__main__':
    # df1 = pd.read_csv(r"C:\Users\11637\PycharmProjects\select-coin\test1.csv")
    # df11 = pd.read_csv(r"C:\Users\11637\PycharmProjects\select-coin\test11.csv")
    # print(df1.equals(df11))
    # df2 = pd.read_csv(r"C:\Users\11637\PycharmProjects\select-coin\test2.csv")
    # df22 = pd.read_csv(r"C:\Users\11637\PycharmProjects\select-coin\test22.csv")
    # print(df2.equals(df22))
    # df2 = pd.read_pickle(get_file_path('data', 'cache', 'all_candle_df_list.pkl'))
    # df22 = pd.read_pickle(r"C:\Users\11637\PycharmProjects\select-coin\all_candle_df_list2.pkl")
    #
    # for x, y in zip(df2, df22):
    #     print(x["symbol"].iloc[0], x.equals(y))
    df = pd.read_pickle(get_file_path('data', 'cache', 'all_factors_df.pkl'))
    print(df.columns)
    pass