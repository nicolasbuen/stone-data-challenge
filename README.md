### Stone Data Challenge - Previsão de TPV

Essa repo foi criada para o Data Challenge da Stone Pagamentos. Os notebooks e a apresentação contidas aqui me garantiram o **primeiro lugar no desafio** e os 15 mil reais da premiação! 

Caso algum dos notebooks não carregue pelo github, fique a vontade para utilizar o [Jupyter Notebook Viewer](https://nbviewer.jupyter.org/github/nicolasbuen/stone-data-challenge/tree/master/)! Não é necessário baixar nada e esse link direciona para os notebooks dessa repo.

Todos os notebooks possuem um resumo do que foi realizado ou do que foi descoberto utilizando eles (no caso do EDA). 

#### Sumário de Arquivos

* O [primeiro notebook](https://github.com/nicolasbuen/stone-data-challenge/blob/master/1.%20Understanding%20and%20Cleaning.ipynb) serviu para dar a primeira olhada nos datasets e realizar o primeiro **pré-processamento e a limpeza dos dados**. Devido ao volume e aos tipos dos dados - mais de 200mil observações contendo muitas strings no cadastrais.csv - ele demora um pouco pra rodar, mas nada muito expressivo (aproximadamente 3min30s). Para facilitar a replicabilidade, criei um script que retorna os mesmos DataFrames que esse notebook - está na pasta `scripts-replicabilidade` e é chamado `cleaning_script.py`.  


* No [segundo notebook](https://github.com/nicolasbuen/stone-data-challenge/blob/master/2.%20EDA.ipynb) foi realizado o **EDA - Exploratory Data Analysis**. Além de realizar alguns procedimentos padrões para a criação do modelo (distribuições, correlação e etc), procurei entender um pouco mais sobre o dataset, realizando a exploração guiado por perguntas que eu criei enquanto limpava-o. Ele foi de extrema importância para a definição do modelo e dos passos a serem feitos no pré-processamento.


* O [terceiro notebook](https://github.com/nicolasbuen/stone-data-challenge/blob/master/3.%20Feature%20Engineering%20e%20Preprocessing.ipynb) contém a **engenharia de features e o pré-processamento dos dados**. Devido ao alto número de features categóricas - algumas delas com centenas de categorias - no dataset original, realizei o pré-processamento com algoritmos tree-based em mente, uma vez que eles lidam melhor com elas. O script que retorna o dataset final desse notebook está na pasta `scripts-replicabilidade` e é chamado `preprocessing.py` - ele recebe os arquivos limpos após o primeiro notebook.


* O [quarto notebook](https://github.com/nicolasbuen/stone-data-challenge/blob/master/4.%20Modelling.ipynb) é onde foi realizado o **treinamento, a validação e a hiperparametrização dos modelos**. Além disso, **é nele que consta o modelo que cria o dataset da previsão - entregável**. Esse notebook toma bastante tempo para rodar! Nele também foi utilizado um dataset criado pelo script `create_predict_df.py`, também na pasta `scripts-replicabilidade`. Ele concatena ao dataset de treinamento (criado pelo `preprocessing.py`) rows com os meses ainda não vistos para todos os IDs. 


* O [quarto notebook versão 2.0](https://github.com/nicolasbuen/stone-data-challenge/blob/master/4.%20Modelling.ipynb) é uma versão pós-entrega. Nela, eu corrijo alguns erros relacionados ao vazamento de informações que ocorreram durante a hiperparametrização e a validação dos modelos no notebook "Modelling". 
