import numpy as np
import datetime
import scipy.stats as st

# Padronização dos Estados

dict_estados = {
                "São Paulo": "SP",
                "Minas Gerais": "MG",
                "Santa Catarina": "SC",
                "Paraná": "PR",
                "Rio de Janeiro":"RJ",
                "Goiás": "GO",
                "Pará": "PA",
                "Bahia": "BA",
                "Mato Grosso": "MT",
                "Rio Grande do Sul": "RS",
                "Maranhão": "MA",
                "Ceará": "CE",
                "Amazonas": "AM",
                "Distrito Federal": "DF",
                "Rondônia": "RO",
                "Espírito Santo": "ES",
                "Sergipe": "SE",
                "Roraima": "RR",
                "Tocantins": "TO",
                "Pernambuco": "PE",
                "Amapá": "AP",
                "Acre": "AC",
                "Paraíba": "PB",
                "Rio Grande do Norte": "RN",
                "Mato Grosso do Sul": "MS",
                "Alagoas": "AL",
                "Piauí": "PI",
                "Sao Paulo": "SP",
                "sc": "SC",
                "Parana": "PR",
                "Rj": "RJ",
                "Sc": "SC"
}

# Estados -> Regiões

dict_regioes = {
                "SP": 'Sudeste',
                "MG": 'Sudeste',
                "RJ": 'Sudeste',
                "ES": 'Sudeste',
                "RS": 'Sul',
                "SC": 'Sul',
                "PR": 'Sul',
                "MT": 'Centro-Oeste',
                "GO": 'Centro-Oeste',
                "DF": 'Centro-Oeste',
                "MS": 'Centro-Oeste',
                "RR": 'Norte',   
                "PA": 'Norte',
                "AM": 'Norte',
                "AP": 'Norte',
                "TO": 'Norte',
                "RO": 'Norte',    
                "AC": 'Norte',       
                "MA": 'Nordeste',
                "CE": 'Nordeste',
                "BA": 'Nordeste',
                "SE": 'Nordeste',
                "PE": 'Nordeste',
                "PB": 'Nordeste',
                "RN": 'Nordeste',
                "AL": 'Nordeste',
                "PI": 'Nordeste',
}

# Ordinal Label StoneCreatedDate

dict_ordinal_dates = {
            '2012-2':0, '2012-3':1, '2012-4':2, '2012-5':3, '2012-6':4, '2012-7':5, '2012-8':6, '2012-9':7,
            '2012-10':8, '2012-11':9, '2012-12':10, '2013-1':11, '2013-2':12, '2013-3':13, '2013-4':14,
            '2013-5':15, '2013-6':16, '2013-7':17, '2013-8':18, '2013-9':19, '2013-10':20, '2013-11':21,
            '2013-12':22, '2014-1':23, '2014-2':24, '2014-3':25, '2014-4':26, '2014-5':27, '2014-6':28,
            '2014-7':29, '2014-8':30, '2014-9':31, '2014-10':32, '2014-11':33, '2014-12':34, '2015-1':35,
            '2015-2':36, '2015-3':37, '2015-4':38, '2015-5':39, '2015-6':40, '2015-7':41, '2015-8':42,
            '2015-9':43, '2015-10':44, '2015-11':45, '2015-12':46, '2016-1':47, '2016-2':48, '2016-3':49,
            '2016-4':50, '2016-5':51, '2016-6':52, '2016-7':53, '2016-8':54, '2016-9':55, '2016-10':56,
            '2016-11':57, '2016-12':58, '2017-1':59, '2017-2':60, '2017-3':61, '2017-4':62, '2017-5':63,
            '2017-6':64, '2017-7':65, '2017-8':66, '2017-9':67, '2017-10':68, '2017-11':69, '2017-12':70,
            '2018-1':71, '2018-2':72, '2018-3':73, '2018-4':74, '2018-5':75, '2018-6':76, '2018-7':77,
            '2018-8':78, '2018-9':79, '2018-10':80, '2018-11':81, '2018-12':82, '2019-1':83, '2019-2':84,
            '2019-3':85, '2019-4':86, '2019-5':87, '2019-6':88, '2019-7':89, '2019-8':90, '2019-9':91, 
            '2019-10':92, '2019-11':93, '2019-12':94, '2020-1':95, '2020-2':96, '2020-3':97, '2020-4':98,
            '2020-5':99, 
            }

# Definindo Funções Apply - cleaning_script

def YmD_to_datetime(date):
    '''
    Recebe um float contendo data em formato YYYYMMDD
    e retorna um datetime no formato padrão do pandas
    '''
    date = str(date)[:-2]
    
    return datetime.datetime.strptime(date, '%Y%m%d').strftime('%m/%d/%Y')

def sample_ci_95(arr, alpha = 0.05):
    '''
    Recebe uma array/series e retorna o intervalo de confiança para a média da população.
    Por padrão, o erro é definido como 0.05.
    
    Considera o fato do std da população não ser conhecido utilizando teste T.
    '''
    confidence_level = 1 - alpha
    degrees_freedom = len(arr)
    sample_mean = np.mean(arr)
    sample_standard_error = st.sem(arr)

    confidence_interval = st.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    
    return confidence_interval


