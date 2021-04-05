# Import libs
import pandas as pd
import numpy as np
import datetime
import scipy.stats as st
from helper import dict_estados, YmD_to_datetime, sample_ci_95

# Import nos datasets
cadastrais = pd.read_csv('data/cadastrais.csv', parse_dates = ['StoneCreatedDate', 'StoneFirstTransactionDate'])
tpv_mensais = pd.read_csv('data/tpv-mensais-treinamento.csv', parse_dates = ['mes_referencia'])

def cleaning_script(cadastrais, tpv_mensais):

    # tpv-mensais.csv
    # Problemas a resolver
    # =========================
    # TPV-mensal < 0
    # Como: Trocar valores < 0 pelo TPV-médio feito pela empresa

    ids_com_tpv_menor_que_zero = tpv_mensais[tpv_mensais['TPV_mensal'] < 0]['id'].unique()

    for empresa_id in ids_com_tpv_menor_que_zero:
        for index, row in tpv_mensais[tpv_mensais['id'] == empresa_id].iterrows():
            if row['TPV_mensal'] < 0:
                    tpv_mensais.iloc[index, 2] = np.mean(tpv_mensais[tpv_mensais['id'] == empresa_id]['TPV_mensal'])

    print("tpv-mensais.csv limpo!")

    # cadastrais.csv
    # Problemas a resolver
    # =========================
    # IDs repetidos
    # IDs que não estão no tpv-mensais
    # NaNs em quase todas as colunas
    # Parse do StoneFirstTransactionDate
    # Contas criadas em 01/01/1990
    # Estados não padronizados - Entrada Manual
    # TPVEstimate com valores errados 

    # Retira os IDs que não estão no tpv-mensais
    ids_repetidos = np.array(cadastrais[cadastrais.duplicated('id', keep = 'first')]['id']) 

    cadastrais = cadastrais[cadastrais['id'].isin(tpv_mensais.id.unique())]

    # NaNs - Resolver com FFill das IDs repetidas

    colunas_com_nan = ['MacroClassificacao', 'segmento', 'sub_segmento', 'Estado']

    cadastrais = cadastrais.sort_values(['id', 'StoneCreatedDate'])

    # Essa função demora um pouco!
    cadastrais[colunas_com_nan] = cadastrais.groupby("id", as_index = False)[colunas_com_nan].fillna(method='ffill')

    # np.where pela eficiencia

    cadastrais['tem_duplicados'] = np.where(cadastrais['id'].isin(ids_repetidos), 1, 0)

    # Transformar tudo em floats (existem alguns 0 em formato de string)

    cadastrais['StoneFirstTransactionDate'] = cadastrais['StoneFirstTransactionDate'].astype(float)

    # Retirar o min (anos 90) - pra poder colocar uma data mais atual se tiver no ID
    stone_created_min = cadastrais['StoneCreatedDate'].min()

    mask_1 = (cadastrais['StoneCreatedDate'] == stone_created_min)
    mask_2 = (cadastrais['tem_duplicados'] == 1)

    cadastrais.loc[(mask_1 & mask_2), 'StoneCreatedDate'] = datetime.date(2100, 1, 1)

    # Exceção - ID c/ 2 rows com anos = 1990

    cadastrais.loc[cadastrais['id'] == 46233, 'StoneCreatedDate'] = stone_created_min

    # StoneFirstTransactionDate = 0 --> 2222200429 || para poder usar o agg `min`

    cadastrais.loc[: , 'StoneFirstTransactionDate'] = np.where(cadastrais.loc[: , 'StoneFirstTransactionDate'] == 0,
                                                                            2222200429,
                                                                            cadastrais.loc[: , 'StoneFirstTransactionDate'])

    date_map = {'StoneCreatedDate': 'min',
                'StoneFirstTransactionDate': 'min'}

    # Criar as colunas de data 

    date_columns = cadastrais.groupby("id", as_index = False).agg(date_map, axis = 1)

    # StoneCreatedDate = 1990 que sobraram -> StoneFirstTransactionDate
    date_columns.loc[date_columns['StoneCreatedDate'] == stone_created_min, 'StoneCreatedDate'] = pd.to_datetime(date_columns.loc[date_columns['StoneCreatedDate'] == stone_created_min, \
                                                                                                                                'StoneFirstTransactionDate'].apply(YmD_to_datetime))

    # Drop duplicados mantendo apenas as últimas

    cadastrais = cadastrais.drop_duplicates('id', keep = 'last').reset_index(drop = True)

    # Drop variáveis de data, merge nas novas -> mantendo assim a verdadeira data de criação e primeira transação

    cadastrais = cadastrais.drop(['StoneCreatedDate', 'StoneFirstTransactionDate'], axis = 1)
    cadastrais = cadastrais.merge(date_columns, on = 'id')

    # Arrumando a coluna StoneFirstTransactionDate

    cadastrais['StoneFirstTransactionDate']  = pd.to_datetime(cadastrais['StoneFirstTransactionDate'].apply(YmD_to_datetime))

    # Padronizando o nome dos estados

    cadastrais['Estado'] = cadastrais['Estado'].replace(dict_estados)

    # FillNA nos restantes

    cadastrais.fillna("Missing Value", inplace = True)

    print("cadastrais.csv limpo!")

    # TPVEstimate c/ valor errado -> 
    # Exportar DataFrame c/ ID, Média do TPV-mensal, o TPV Estimate, Diff(mensal - estimate) e CI

    # Cria um DataFrame com a média dos TPV_mensais e o TPVEstimate por ID

    estimate_and_mean_tpv = tpv_mensais.groupby('id')\
                                                .mean()\
                                                .merge(cadastrais[['id', 'TPVEstimate']], on = 'id')


    # Cria um DataFrame com o intervalo de confiança da média dos TPV_mensais por ID

    mean_ci = tpv_mensais[['id', 'TPV_mensal']]\
                            .groupby('id')['TPV_mensal']\
                            .apply(sample_ci_95)\
                            .apply(pd.Series)\
                            .rename(columns={0:'TPV_medio_CI_LOW',
                                            1:'TPV_medio_CI_HIGH'})


    # Reorganiza as variáveis, merge em ambos os DFs e cria novas variáveis 'is_estimate_in_CI' e 'diff'

    estimate_and_mean_tpv = estimate_and_mean_tpv[['id', 'TPVEstimate', 'TPV_mensal']]

    estimate_and_mean_tpv['diff'] = estimate_and_mean_tpv['TPVEstimate'] - estimate_and_mean_tpv['TPV_mensal']

    estimate_and_mean_tpv = estimate_and_mean_tpv.merge(mean_ci, on = 'id')

    estimate_and_mean_tpv['is_estimate_in_CI'] = np.where(((estimate_and_mean_tpv['TPVEstimate'] >= estimate_and_mean_tpv['TPV_medio_CI_LOW']) &\
                                                        (estimate_and_mean_tpv['TPVEstimate'] <= estimate_and_mean_tpv['TPV_medio_CI_HIGH'])),
                                                        1, 0)                                                    
    print("estimativa-e-tpv-medio-ci.csv foi criado!")

    return cadastrais, tpv_mensais, estimate_and_mean_tpv

cadastrais, tpv_mensais, estimate_and_mean_tpv = cleaning_script(cadastrais, tpv_mensais)

# Exporta p/ CSV

estimate_and_mean_tpv.to_csv("data/estimativa-e-tpv-medio-ci.csv")
tpv_mensais.to_csv("data/tpv-mensais-limpo.csv")
cadastrais.to_csv("data/cadastrais-limpo.csv")

print("DataFrames exportados em CSV para a pasta!")