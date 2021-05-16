# Import libs

# Wrangling e Analysis
import pandas as pd
import numpy as np

# Models
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def model_validation(df_past: pd.DataFrame, df_future,
                     params: dict, model: str, mes_inicial=3, mes_final=7,
                     plot_cat_boost=False, is_validation=True, is_objective_function=False):
    
    '''
    Função que realiza a validação dos resultados obtidos através da hiperparametrização. 
    Caso is_validation == False, a função entenderá que estará prevendo meses ainda não conhecidos e 
    não realizará cálculo de scores.
    
    Toma como parâmetros:
    df_past: um DataFrame pré-processado com "TPV_mensal" = 0 nos meses em que a previsão será feita
    df_future: um DataFrame pré-processado com o TPV_mensal verdadeiro (meses já vistos)
    params: parâmetros obtidos através da hiperparametrização para o modelo.
    model: string com algum dos dois modelos - "catboost" , "rf" ou "lgbm"
    mes_inicial: o mês que a previsão começa - default = 3 (Março)
    mes_final: o último mês em que será realizada a previsão - default = 7 (Julho)
    plot_cat_boost: boolean que dita os plots train-test do catboost
    is_validation: boolean definindo se a função servirá para validação ou previsão
    is_objective_function: boolean definindo se a função servirá para uso com a hiperparâmetrização do optuna
    '''
    
    if is_validation:
        assert type(df_future) == pd.DataFrame
    
    # Inicializa listas de score e lista de dataframes (gerar predict)
    mean_error = []
    mean_sq_error = []
    df_list = []
    
    # Não realizar alterações ao df original (reutilização)
    df = df_past.copy()
    
    for mes in range(mes_inicial, mes_final+1):
        
        # Treina com todos os meses até o 'mes' não incluso
        train = df[df['mes_referencia'] < np.datetime64(f'2020-0{mes}')]
        
        # Mês Para Previsão
        val = df[(df['Ano'] == 2020) & (df['Mês'] == mes)]
    
        # Separa os ids (algumas empresas possuem meses faltantes)
        ids = val.id.reset_index(drop=True)

        if model == 'rf':
            val = val.fillna(0)
            train = train.fillna(0)
    
        # Train-Test-Split manual
        X_train = train.drop(['TPV_mensal', 'mes_referencia'], axis=1)
        X_future = val.drop(['TPV_mensal', 'mes_referencia'], axis=1)

        y_train = train['TPV_mensal'].values
        y_true = val['TPV_mensal'].values
        
        # Caso seja utilizado para validação
        if is_validation:
            # criando y_true utilizando os dados do "futuro"
            mask_1 = df_future['Mês'] == mes
            mask_2 = df_future['Ano'] == 2020
            mask_3 = df_future.id.isin(val.id.unique())

            df_future_masked = df_future[mask_1 & mask_2 & mask_3]
            y_true = df_future_masked['TPV_mensal']
        
            # Separa os ids do y_true (algumas empresas possuem meses faltantes)
            ids_df_future_masked = df_future_masked.id.unique()
            mask_pred = X_future.id.isin(ids_df_future_masked)
        
        # == Começando o Modelo == 
        
        # CatBoost
        if model == 'catboost':
            
            model_trained = cb.CatBoostRegressor(logging_level='Silent',
                                                 **params)
            
            if is_validation:
                model_trained.fit(X_train, y_train, plot=plot_cat_boost,
                                eval_set=[(X_future[mask_pred], y_true)],
                                cat_features=['id', 'MCC', 'MacroClassificacao',
                                                'sub_segmento', 'persona', 'porte',
                                                'tipo_documento', 'Estado', 'StoneCreatedDate',
                                                'Região'],
                                early_stopping_rounds=300)
            
            else:
                model_trained.fit(X_train, y_train, plot=plot_cat_boost,
                                cat_features=['id', 'MCC', 'MacroClassificacao',
                                                'sub_segmento', 'persona', 'porte',
                                                'tipo_documento', 'Estado', 'StoneCreatedDate',
                                                'Região'],
                                early_stopping_rounds=300)
            
        # LightGBM
        elif model == 'lgbm':
            
            train_data_lgbm = lgb.Dataset(X_train,
                                         label=y_train,
                                         free_raw_data=False)

            test_data_lgbm = lgb.Dataset(X_future[mask_pred],
                                         label=y_true,
                                         reference=train_data_lgbm,
                                         free_raw_data=False)

            model_trained = lgb.train(params, train_data_lgbm,                     
                                      valid_sets=[train_data_lgbm, test_data_lgbm],
                                      verbose_eval=0)

        # Random Forest
        elif model == 'rf':
            model_trained = RandomForestRegressor()
            model_trained.fit(X_train, y_train)

        # Previsão -> Adiciona à lista de previsões
        pred = model_trained.predict(X_future)
        series_pred = pd.Series(pred)
        df_list.append(pd.DataFrame([ids, series_pred]).T.rename(columns={'Unnamed 0': mes}))

        # Métricas - validação
        if is_validation:
            predict_future_masked = model_trained.predict(X_future[mask_pred])
            sqr_error = mean_squared_error(y_true, predict_future_masked)
            error = mean_absolute_error(y_true, predict_future_masked)
            mean_sq_error.append(sqr_error)
            mean_error.append(error)

            if not is_objective_function:
                print('Mês %d - Erro Quadrado %.5f\t- Erro Absoluto %.5f ' % (mes, sqr_error, error))

        # Atualizando o dataframe com as previsões
        mask_month = df['Mês'] == mes 
        mask_year =  df['Ano'] == 2020

        df.loc[mask_month & mask_year, 'TPV_mensal'] = np.array(series_pred)
        
        # Atualizando as features do modelo utilizando os dados previstos
        df.loc[mask_month & mask_year, 'tpv_ultimo_mes'] = df.groupby(['id'])['TPV_mensal'].shift()
        df.loc[mask_month & mask_year, 'diff_ultimo_mes'] = df.groupby(['id'])['tpv_ultimo_mes'].diff()

        for i in range(1, 12):
            df.loc[mask_month & mask_year, f'tpv_ultimo-{i}_mes'] = df.groupby(['id'])['TPV_mensal'].shift(i+1)
            df.loc[mask_month & mask_year, f'diff_ultimo-{i}_mes'] = df.groupby(['id'])[f'tpv_ultimo-{i}_mes'].diff()
            
    if is_validation and not is_objective_function:
        print('Média Erro Quadrado = %.5f\t- Média Erro Absoluto %.5f' % \
              (np.mean(mean_sq_error), np.mean(mean_error)))

    if is_objective_function:
        return np.mean(mean_error)

    else:
        predictions = df_list[0].copy()

        for df_p in df_list[1:]:
            predictions = predictions.merge(df_p, on='id')
    
        return df, model_trained, predictions
