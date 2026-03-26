import pandas as pd




#Removendo outliers

def iqr_bounds(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    return low, high

def remove_outliers_iqr(df, column, category_col=None):

    # valida se coluna existe
    if column not in df.columns:
        raise ValueError(f"Coluna '{column}' não encontrada. Colunas disponíveis: {list(df.columns)}")

    if category_col and category_col not in df.columns:
        raise ValueError(f"Coluna de categoria '{category_col}' não encontrada.")

    if category_col:
        result = pd.DataFrame()

        for category in df[category_col].dropna().unique():
            subset = df[df[category_col] == category]
            low, high = iqr_bounds(subset[column])
            mask = (subset[column] >= low) & (subset[column] <= high)
            result = pd.concat([result, subset[mask]])

        return result.reset_index(drop=True)

    else:
        low, high = iqr_bounds(df[column])
        mask = (df[column] >= low) & (df[column] <= high)

        return df[mask].reset_index(drop=True)

#Tratando os valores nulos

def missing_values (df):

    cols_num = df.select_dtypes(include='number').columns
    for col in cols_num:
        df[col] = df[col].fillna(df[col].median()) 
    
    return df


#Balanceando as classes

def mapping_data (df):

    mapping = {
        'Excelente': 'Boa',
        'Boa': 'Boa',
        'Moderada': 'Moderada',
        'Ruim': 'Ruim',
        'Muito Ruim': 'Ruim'
    }

    df = df.copy()
    
    df['Qualidade_Ambiental'] = df['Qualidade_Ambiental'].map(mapping)
    
    return (df)

#pipeline transforação

def preprocessing_pipeline(df):
    
    df = df.copy()
    df['Pressao_Atm'] = pd.to_numeric(df['Pressao_Atm'], errors='coerce')

    #Valores nulos
    df = missing_values(df)
    
    #outliers
    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if col != 'Qualidade_Ambiental':  
            df = remove_outliers_iqr(df, col, 'Qualidade_Ambiental')

    #Mapping
    df = mapping_data (df) 
    
    # Criar caminho de saída na mesma pasta
    df.to_csv("processed_data.csv")
    
    return df