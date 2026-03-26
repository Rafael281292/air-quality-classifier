import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Criando pasta para as figuras
base_dir = os.path.dirname(os.path.abspath(__file__))
pasta = os.path.join(base_dir, "analise_exploratoria")
os.makedirs(pasta, exist_ok=True)

#Definindo uma função

def anal_exp (df):
    categorical = df.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    numerical = df.select_dtypes(include='number').columns.tolist()
    print(df.head(5))
    print(20*"--")
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    print(20*"--")
    print("\nValores únicos por coluna:")
    print(df.nunique())
    print(20*"--")
    print("Resumo dataset:")
    print(df.describe())
    print(df.info)
    
    #Distribuição de classes
    
    def distri_class(df, categorical, pasta):
        for col in categorical:
            freq = df[col].value_counts()

            plt.figure(figsize=(6,4))
            freq.plot(kind='bar')

            plt.title(f"Frequência de {col}")
            plt.ylabel("Frequência")
            plt.xlabel(col)

            plt.savefig(f"{pasta}/frequencia_classes.png", dpi=300, bbox_inches='tight')
                         
            plt.close()
            
    distri_class (df, categorical, pasta)

    #Boxplots
    
    def box_plot (df, numerical, pasta):
        for col in numerical:
            plt.figure(figsize=(6,4))
        
            sns.boxplot(
                data=df,
                x='Qualidade_Ambiental',
                y=col,
                palette='pastel'
            )
        
            plt.title(f"{col} vs Qualidade_Ambiental")
        
            plt.savefig(f"{pasta}/boxplot_{col}.png", dpi=300, bbox_inches='tight')
            plt.close()
         
    box_plot (df, numerical, pasta)
    
    # Satterplot
    
    def scatter_plot (df, numerical, pasta):
        for col in numerical:
    
            plt.figure(figsize=(6,4))       
        
            sns.scatterplot(
                data=df,
                x='Qualidade_Ambiental',
                y=col,
                hue='Qualidade_Ambiental',
                palette='RdYlGn'
            )
            plt.title(f"{col} vs Qualidade_Ambiental")
            plt.savefig(f"{pasta}/scatter_{col}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
    scatter_plot (df, numerical, pasta)
    
    #Matriz de correlação
    
    def corr_mat (df, numerical, pasta):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical].corr(), annot=True, cmap='Blues', fmt='.2f')
        plt.title(f'Matriz de Correlação')

        
        plt.savefig(f"{pasta}/Matriz_correlação.png", dpi=300, bbox_inches='tight')
        plt.close()
    corr_mat (df, numerical, pasta)   
    
    #Pair plot
    
    def pair_plot (df, categorical, pasta):
      
        pair = sns.pairplot(df, hue = categorical[0])
          
        pair.fig.suptitle(f'Pairplot - Ambiental', y=1.02)
        caminho = os.path.join(pasta, f"pairplot_{categorical[0].replace(' ', '_')}.png")
        pair.savefig(caminho, bbox_inches="tight")
        plt.close('all')        
    pair_plot (df, categorical, pasta)
    
    return (df)

 


