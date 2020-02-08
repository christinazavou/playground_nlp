

def chunk_dataframe_serial(df, chunk_size):
    df_len = df.shape[0]
    for i in range(0, df_len, chunk_size):
        yield df.iloc[i:i + chunk_size]
