# Import libs - wrangling e analysis
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Import libs - Stats
from scipy.stats import boxcox

from helper import dict_regioes, dict_ordinal_dates

# Import nos datasets
cadastrais = pd.read_csv('data/cadastrais-limpo.csv', parse_dates = ['StoneCreatedDate', 'StoneFirstTransactionDate'])
cadastrais.drop("Unnamed: 0", axis = 1, inplace = True)

tpv_mensais = pd.read_csv('data/tpv-mensais-limpo.csv', parse_dates = ['mes_referencia'])
tpv_mensais.drop("Unnamed: 0", axis = 1, inplace = True)

def preprocessing(cadastrais, tpv_mensais):

    # Features Categóricas
    # =======================
    # Redução do número de categorias nas variáveis MCC e sub_segmentos
    # Adição de colunas 'is_missing' ao lado de categorias que possuíam valores missing.
    # Adição de coluna 'Região', agrupando os estados
    # Label-Encoding em features nominais
    # Ordinal-Encoding em features ordinais

    # Features Datetime
    # =======================
    # Adição de feature "tempo entre criação de conta e primeira transação"
    # Encoding no mês-ano da StoneCreatedDate e drop da StoneFirstTransaction
    # Decompose do mes-referencia em mês, trimestre e ano.

    # Features Contínuas
    # =======================
    # Drop na TPV-Estimado
    # Criação de Lags e Diffs dos TPV-mensais anteriores
 

    # Porte Encoding -> Ordinal
    porte_encoding = {
                '0-2.5k': 1,
                '2.5k-5k': 2,
                '5k-10k': 3,
                '10k-25k': 4,     
                '25k-50k': 5,    
                '50k-100k': 6,   
                '100k-500k': 7,   
                '500k+': 8,    
                }

    cadastrais['porte'] = cadastrais['porte'].replace(porte_encoding)

    # Persona Encoding -> Ordinal-ish
    cadastrais['persona'] = cadastrais['persona'].replace({'Digital - Micro':'Micro Empreendedor'})

    persona_encoding = {
                'Micro Empreendedor': 1,
                'Pequeno Empreendedor': 2,
    
                'SMB - Pequeno Porte e Ticket Baixo': 5,
                'SMB - Pequeno Porte e Ticket Medio': 6,     
                'SMB - Pequeno Porte e Ticket Alto': 7,
    
                'Outro': 8,
    
                'SMB - Medio Porte e Ticket Baixo': 10,   
                'SMB - Medio Porte e Ticket Medio': 11,   
                'SMB - Medio Porte e Ticket Alto': 12,
    
                'SMB - Grande Porte e Ticket Baixo': 15,   
                'SMB - Grande Porte e Ticket Medio': 16,   
                'SMB - Grande Porte e Ticket Alto': 17,  
                }

    cadastrais['persona'] = cadastrais['persona'].replace(persona_encoding)

    # Criação de Feature -> Tempo entre criação de Conta e Transação
    cadastrais['diff_FirstTransaction_Created'] = (cadastrais['StoneFirstTransactionDate'] - \
                                                   cadastrais['StoneCreatedDate']).dt.days

    diff_median = cadastrais['diff_FirstTransaction_Created'].median()

    cadastrais.loc[:, 'diff_FirstTransaction_Created'] = np.where(cadastrais['diff_FirstTransaction_Created'] < 0,
                                                                  diff_median, cadastrais['diff_FirstTransaction_Created'])

    # StoneCreatedDate -> MM-YYYY, drop FirstTransaction
    cadastrais.loc[:, 'StoneCreatedDate'] = cadastrais['StoneCreatedDate'].apply(lambda x: f'{x.year}-{x.month}')
    
    cadastrais.drop(['StoneFirstTransactionDate', 'TPVEstimate'], axis = 1, inplace = True)

    cadastrais.loc[:, 'StoneCreatedDate'] = cadastrais['StoneCreatedDate'].replace(dict_ordinal_dates)

    # Normalização da Feature Criada
    diff_trans_created_norm, _ = boxcox(cadastrais['diff_FirstTransaction_Created'].add(0.5))

    cadastrais.loc[:, 'diff_FirstTransaction_Created'] = diff_trans_created_norm

    # Criação de Feature -> is_missing para valores NaN
    for col in ['MacroClassificacao', 'segmento', 'sub_segmento', 'Estado']:
        cadastrais[f'is_missing_{col}'] = np.where(cadastrais[col] == 'Missing Value',
                                                    1, 0)

    cadastrais = cadastrais[['id', 'MCC', 'MacroClassificacao', 'is_missing_MacroClassificacao',
                            'segmento','is_missing_segmento', 'sub_segmento', 'is_missing_sub_segmento',
                            'persona', 'porte', 'tipo_documento', 'Estado', 'is_missing_Estado',
                            'tem_duplicados', 'StoneCreatedDate', 'diff_FirstTransaction_Created']]

    # Criação de Feature -> Região da Empresa
    cadastrais['Região'] = cadastrais['Estado'].replace(dict_regioes)

    # Redução do Número de Categorias
    for col, threshold in zip(['MCC', 'sub_segmento'], [0.0005, 0.005]):
    
        value_count = cadastrais[col].value_counts(normalize = True)

        cats_below_threshold = np.array(value_count[value_count < threshold].index)

        cadastrais.loc[:, col] = ['Outro' if x in cats_below_threshold else x for x in cadastrais[col]]

    # Label Encoding das Categóricas Restantes
    for col in ['MCC', 'MacroClassificacao', 'segmento', 'sub_segmento', 'tipo_documento', 'Estado', 'Região']:
    
        cadastrais.loc[:, col] = cadastrais[col].apply(lambda x: str(x))
        le = LabelEncoder()
        le.fit(cadastrais[col])
        cadastrais.loc[:, col] = le.transform(cadastrais[col])

    # Criação de Features -> Decomposição da Data
    tpv_mensais = tpv_mensais.sort_values(['id','mes_referencia'])

    tpv_mensais['Mês'] = tpv_mensais['mes_referencia'].dt.month
    tpv_mensais['Trimestre'] = tpv_mensais['mes_referencia'].dt.quarter
    tpv_mensais['Ano'] = tpv_mensais['mes_referencia'].dt.year

    # Criação de Features -> Lag e Diff do TPV-Mensal
    tpv_mensais['tpv_ultimo_mes'] = tpv_mensais.groupby(['id'])['TPV_mensal'].shift()
    tpv_mensais['diff_ultimo_mes'] = tpv_mensais.groupby(['id'])['tpv_ultimo_mes'].diff()

    for i in range(1, 11):
        tpv_mensais[f'tpv_ultimo-{i}_mes'] = tpv_mensais.groupby(['id'])['TPV_mensal'].shift(i)
        tpv_mensais[f'diff_ultimo-{i}_mes'] = tpv_mensais.groupby(['id'])[f'tpv_ultimo-{i}_mes'].diff()

    # DataFrame Modelo
    df_modelo = tpv_mensais.merge(cadastrais, on = 'id')

    return df_modelo

