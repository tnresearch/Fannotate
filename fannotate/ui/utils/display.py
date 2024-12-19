import pandas as pd

def process_df_for_display(df, top_n=5):
    """
    Purpose: Formats a DataFrame for display in the UI by truncating long text fields for better readability.
    """
    if df is None:
        return None
    try:
        if isinstance(df, pd.DataFrame):
            df_display = df.copy()
        else:
            df_display = pd.DataFrame(df.value if hasattr(df, 'category') else df)
        
        if 'text' in df_display.columns:
            df_display['text'] = df_display['text'].astype(str).apply(
                lambda x: x[:25] + '...' if len(x) > 25 else x)
        
        for column in df_display.columns:
            if column != 'text' and df_display[column].dtype == 'object':
                df_display[column] = df_display[column].astype(str).apply(
                    lambda x: x[:500] + '...' if len(x) > 500 else x)
        
        return df_display.head(top_n)
    except Exception as e:
        print(f"Error processing DataFrame: {e}")
        return None

def clean_column_name(name):
    """
    Purpose: Sanitizes column names by removing special characters and spaces.
    """
    if isinstance(name, list):
        name = "".join(name)
    return name.strip().replace('[','').replace(']','').replace("'", '').replace(" ", '_') 