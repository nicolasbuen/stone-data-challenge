# Wrangling e Analysis
import pandas as pd
import numpy as np


# Importando DataFrames
cadastrais = pd.read_csv('data/cadastrais_modelo.csv')
cadastrais.drop("Unnamed: 0", axis = 1, inplace = True)

tpv_mensais = pd.read_csv('data/tpv_mensais_modelo.csv', parse_dates = ['mes_referencia'])
tpv_mensais.drop("Unnamed: 0", axis = 1, inplace = True)


def create_dataframe_for_prediction(cadastrais, tpv_mensais):

    tpv_mensais = tpv_mensais\
                    .sort_values(
                    ['mes_referencia', 'id'])\
                    [['id', 'mes_referencia', 'TPV_mensal', 'Mês', 'Trimestre', 'Ano']]

    row_list = []

    # Cria novas rows c/ meses ainda não vistos
    for id in tpv_mensais.id.unique():
        for mes in [8, 9, 10, 11, 12]:
            if mes/10 >= 1:
                new_row = {'id': id,
                        'mes_referencia': np.datetime64(f'2020-{mes}-30'),
                        'TPV_mensal': 0,
                        'Mês': mes,
                        'Trimestre': 4,
                        'Ano': 2020}
                
                row_list.append(new_row)
                
            else:
                new_row = {'id': id,
                        'mes_referencia': np.datetime64(f'2020-0{mes}-30'),
                        'TPV_mensal': 0,
                        'Mês': mes,
                        'Trimestre': 3,
                        'Ano': 2020}
                
                row_list.append(new_row)

    tpv_ult_meses = pd.DataFrame.from_dict(row_list)

    tpv_mensais = pd.concat([tpv_mensais, tpv_ult_meses], axis = 0)

    tpv_mensais = tpv_mensais.sort_values(['id','mes_referencia'])

    # Cria Lags e Diffs para o TPV-mensal, ajuda bastante em previsão de séries temporais
    tpv_mensais['tpv_ultimo_mes'] = tpv_mensais.groupby(['id'])['TPV_mensal'].shift()
    tpv_mensais['diff_ultimo_mes'] = tpv_mensais.groupby(['id'])['tpv_ultimo_mes'].diff()

    for i in range(1, 11):
        tpv_mensais[f'tpv_ultimo-{i}_mes'] = tpv_mensais.groupby(['id'])['TPV_mensal'].shift(i)
        tpv_mensais[f'diff_ultimo-{i}_mes'] = tpv_mensais.groupby(['id'])[f'tpv_ultimo-{i}_mes'].diff()

    # Altera 0s pra NaN, uma vez que os algoritmos lidam melhor com ele
    tpv_mensais.iloc[:, 6:28] = tpv_mensais.iloc[:, 6:28].replace(0, np.nan)

    # merge no tpv_mensais com o cadastrais
    df_treino_final = tpv_mensais.merge(cadastrais, on = 'id')

    return df_treino_final

df_treino_final = create_dataframe_for_prediction(cadastrais, tpv_mensais)

df_treino_final.to_csv('data/df_treino_final.csv')
