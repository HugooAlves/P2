● Nome do projeto
RELATÓRIO P2 – DETECÇÃO DE DEFAULT DE CLIENTES DE CARTÃO DE CRÉDITO

● Integrantes e RAs
Mateus Rodrigues (RA: 2037922)

Hugo Alves (RA: 2045165)

Leonardo Adorno (RA: 2034220)

Instituição: Universidade de Marília – UNIMAR

Disciplina: Inteligência Artificial – Classificadores

● Descrição do problema
A inadimplência (default) de cartões de crédito é um dos maiores problemas enfrentados por bancos e instituições financeiras. Quando um cliente não paga a fatura, a instituição sofre um prejuízo financeiro direto. O desafio é conseguir analisar o histórico financeiro e demográfico de milhares de clientes e identificar, de forma automatizada, quem apresenta um alto risco de não honrar os pagamentos no mês seguinte.

● Objetivo do projeto
Desenvolver, comparar e avaliar modelos de Machine Learning capazes de prever a probabilidade de inadimplência de um cliente (variável default.payment.next.month). Além disso, o projeto buscou corrigir falhas metodológicas de versões anteriores (como o Data Leakage), aplicar técnicas de balanceamento de dados e entregar uma solução prática através de uma aplicação interativa.

● Dataset utilizado
O dataset utilizado foi o Default of Credit Card Clients, que contém dados de aproximadamente 30.000 clientes. Ele é composto por 25 colunas, incluindo:

Dados Cadastrais/Demográficos: Limite de crédito (LIMIT_BAL), Idade (AGE), Sexo (SEX), Escolaridade (EDUCATION) e Estado Civil (MARRIAGE).

Histórico de Comportamento (Últimos 6 meses): Status de atraso das faturas (PAY_0 a PAY_6).

Valores Financeiros (Últimos 6 meses): Valor das faturas (BILL_AMT1 a BILL_AMT6) e valores que foram efetivamente pagos (PAY_AMT1 a PAY_AMT6).

● Tipo de problema de Machine Learning
Trata-se de um problema de Aprendizado Supervisionado de Classificação Binária.

Supervisionado: Porque o modelo aprende a partir de dados históricos que já possuem o "gabarito" (sabemos quem pagou e quem atrasou).

Classificação Binária: Porque a resposta final possui apenas duas categorias possíveis: 0 (adimplente / vai pagar) ou 1 (inadimplente / vai dar calote).

● Metodologia
A metodologia do projeto seguiu rigorosamente as boas práticas de Ciência de Dados para evitar o Data Leakage (vazamento de dados):

Análise Exploratória (EDA): Inspeção de nulos, tipos de dados, distribuição da classe alvo e análise de correlação através de gráficos (countplot e heatmap).

Engenharia de Recursos (Feature Engineering): Criação das variáveis TOTAL_BILL (soma das faturas), TOTAL_PAY (soma dos pagamentos) e PAY_RATIO (razão de pagamento).

Divisão dos Dados: Divisão estratificada (mantendo a proporção de inadimplentes) em 3 conjuntos independentes: Treino (70%), Validação (15%) e Teste (15%).

Pré-processamento (Escalonamento): Aplicação do StandardScaler para normalizar as variáveis numéricas (calculado apenas no treino e replicado na validação/teste).

Balanceamento: Uso da técnica SMOTE no conjunto de treino para criar dados sintéticos da classe minoritária (inadimplentes) e equilibrar o aprendizado.

Validação Cruzada: Uso de StratifiedKFold com 5 splits no treino para avaliar a estabilidade dos modelos.

● Modelos treinados
Foram treinados e testados três algoritmos de famílias diferentes:

Regressão Logística (Logistic Regression): Modelo linear estatístico clássico, usado como ponto de partida (baseline).

Floresta Aleatória (Random Forest Classifier): Modelo de conjunto (Ensemble) baseado em múltiplas árvores de decisão atuando em paralelo.

XGBoost (XGBClassifier): Modelo de Gradient Boosting de alta performance que cria árvores sequenciais corrigindo os erros das anteriores.

● Modelo final escolhido
O modelo final escolhido foi o Random Forest. Ele foi selecionado por apresentar a maior consistência geral e o melhor equilíbrio entre poder de previsão e capacidade de generalização com dados inéditos.

● Métricas de avaliação
Como a base de dados é altamente desbalanceada, utilizar apenas a Acurácia seria um erro. Por isso, foram adotadas métricas mais rigorosas:

Acurácia (Accuracy): Taxa geral de acertos.

Precisão (Precision): Quantidade de inadimplentes preditos corretamente dividida pelo total de alertas gerados (evita falsos alarmes).

Recall (Sensibilidade): Capacidade do modelo de encontrar os inadimplentes reais dentro da base.

F1-Score: Média harmônica entre Precisão e Recall.

AUC-ROC (Área sob a Curva ROC): Mede a capacidade do modelo de separar e distinguir as duas classes.

● Principais resultados
Os modelos baseados em árvores superaram drasticamente a Regressão Logística. Os resultados finais no conjunto de teste foram:

Random Forest: Acurácia de 78,06% | Precisão: 50,39% | Recall: 57,53% | F1-Score: 53,72% | AUC-ROC: 0,7702

XGBoost: Acurácia de 78,33% | Precisão: 51,03% | Recall: 52,00% | F1-Score: 51,51% | AUC-ROC: 0,7626

Regressão Logística: Acurácia de 68,80% | Precisão: 37,68% | Recall: 62,65% | F1-Score: 47,05% | AUC-ROC: 0,7135

Análise de Importância: O gráfico de Feature Importance provou que o histórico recente de atrasos (PAY_0 e PAY_2) são os fatores mais determinantes para prever o risco.

● Estrutura dos arquivos
Para garantir que o projeto funcione e seja colocado em produção, a estrutura gerada conta com os seguintes arquivos principais:

notebook.ipynb: Arquivo contendo todo o código de análise, tratamento e treino dos modelos.

modelo_final.joblib: O cérebro do modelo Random Forest treinado e salvo em disco.

scaler.joblib: O padronizador de escala salvo para ser usado nos dados que entrarem no app.

app.py: Código em Python que cria a interface visual do Streamlit.

● Tecnologias utilizadas
Linguagem: Python

Manipulação e Matemática: Pandas e NumPy

Gráficos e Visualização: Matplotlib e Seaborn

Machine Learning e Pré-processamento: Scikit-Learn (sklearn)

Algoritmo de Boosting Avançado: XGBoost

Balanceamento de Dados: Imbalanced-Learn (imblearn / SMOTE)

Persistência (Salvar o modelo): Joblib

Interface Web e Publicação: Streamlit

● Instruções para executar o notebook
Certifique-se de ter o Python e o Jupyter Notebook (ou VS Code / Google Colab) instalados.

Instale as bibliotecas necessárias rodando no terminal:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib

Abra o arquivo do notebook e certifique-se de que o arquivo do dataset (.csv) está na mesma pasta.

Execute as células sequencialmente de cima para baixo (Menu Kernel -> Restart & Run All).

● Instruções para executar o app Streamlit
No terminal do seu computador, navegue até a pasta onde estão os arquivos app.py, modelo_final.joblib e scaler.joblib.

Instale o Streamlit caso ainda não o tenha: pip install streamlit

Execute o comando: streamlit run app.py

Uma aba no seu navegador web padrão abrirá automaticamente exibindo a aplicação interativa, onde é possível preencher os dados de um cliente fictício e simular a previsão de inadimplência em tempo real.

● Link do app publicado
https://8502-m-s-kkb-ase1a2-ri79vgwah349-a.asia-east1-2.prod.colab.dev)).

● Limitações
Dados Estáticos: O modelo depende estritamente do comportamento contido nesse histórico de dados de crédito e pode precisar de reciclagem (re-treino) se o cenário econômico ou o perfil dos clientes mudar drasticamente ao longo do tempo.

Atrasos Recentes Dominantes: Como as variáveis PAY_0 e PAY_2 têm um peso massivo, o modelo pode ter dificuldades em classificar corretamente clientes que possuem faturas altíssimas (BILL_AMT), mas que ainda não registraram atrasos formais no sistema.

● Conclusão
O projeto atingiu com sucesso seus objetivos ao evoluir a qualidade metodológica da versão anterior. A correção do Data Leakage garantiu resultados honestos e realistas. O uso do SMOTE contornou o desafio do desbalanceamento de classes, permitindo que o Random Forest fosse treinado com eficiência para identificar o perfil de risco de inadimplência. Por fim, a criação do aplicativo em Streamlit provou como a Inteligência Artificial pode sair do papel e se transformar em uma ferramenta de tomada de decisão prática e acessível para o mercado financeiro e de gestão de riscos.
