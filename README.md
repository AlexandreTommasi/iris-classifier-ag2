# 🌸 Classificador de Flores Iris - AG2

Sistema de classificação de flores Iris utilizando Machine Learning com Decision Tree Classifier.

---

## 👥 Equipe

- **Alexandre Tommasi**
- **Davi Augusto**

---

## 📊 Dataset

O projeto utiliza o famoso **Iris Dataset**:

- **Arquivo:** `iris.csv`
- **Total de amostras:** 150 flores
- **Número de classes:** 3 espécies
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica
- **Features (4 características):**
  - Comprimento da sépala (sepal length)
  - Largura da sépala (sepal width)
  - Comprimento da pétala (petal length)
  - Largura da pétala (petal width)

---

## 🔧 Tratamento de Dados

### Conversão de Espécies: String → Integer

Para permitir o processamento pelo modelo de Machine Learning, as espécies são convertidas de strings para valores inteiros usando o método `.replace()`:

```python
df['species'] = df['species'].replace({
    'Iris-setosa': 1,
    'Iris-versicolor': 2,
    'Iris-virginica': 3
}).astype('int64')
```

**Mapeamento aplicado:**
- `Iris-setosa` → `1`
- `Iris-versicolor` → `2`
- `Iris-virginica` → `3`

---

## 🤖 Modelo

### Decision Tree Classifier

**Características do modelo:**
- **Algoritmo:** Decision Tree (Árvore de Decisão)
- **Framework:** scikit-learn
- **Divisão dos dados:** 80% treino / 20% teste
- **Configuração de split:**
  - `test_size=0.2`
  - `random_state=42`
  - `shuffle=True`
  - `stratify=y` (mantém proporção das classes)

---

## 📁 Estrutura do Projeto

```
iris-classifier-ag2/
│
├── data_loader.py          # Carregamento e preparação dos dados
├── model_trainer.py        # Treinamento do modelo Decision Tree
├── model_evaluator.py      # Avaliação e métricas do modelo
├── classifier_interface.py # Interface interativa para classificação
├── main.py                 # Pipeline completo integrado
├── iris.csv                # Dataset original
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação (este arquivo)
```

### Descrição dos arquivos Python:

| Arquivo | Descrição |
|---------|-----------|
| `data_loader.py` | Carrega o CSV, converte espécies para inteiros, explora dados, divide em treino/teste e salva dados preparados |
| `model_trainer.py` | Carrega dados preparados, treina o Decision Tree, avalia preliminarmente e salva o modelo |
| `model_evaluator.py` | Carrega modelo treinado, gera métricas detalhadas e cria matriz de confusão |
| `classifier_interface.py` | Interface interativa para classificar novas flores inserindo medidas manualmente |
| `main.py` | Orquestra todo o pipeline executando todos os módulos em sequência |

---

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passos para instalação

1. **Clone o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd iris-classifier-ag2
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv venv
   ```

3. **Ative o ambiente virtual:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Como Usar

### Executar o pipeline completo (recomendado)

```bash
python main.py
```

Este comando executa automaticamente todas as etapas do projeto em ordem.

---

### Executar módulos individualmente

Se preferir executar cada etapa separadamente:

1. **Carregar e preparar dados:**
   ```bash
   python data_loader.py
   ```

2. **Treinar o modelo:**
   ```bash
   python model_trainer.py
   ```

3. **Avaliar o modelo:**
   ```bash
   python model_evaluator.py
   ```

4. **Usar a interface interativa:**
   ```bash
   python classifier_interface.py
   ```

---

## ✅ Etapas Implementadas

- [x] **Etapa 1:** Carregamento do dataset a partir do arquivo CSV
- [x] **Etapa 2:** Conversão das espécies de String para Integer
- [x] **Etapa 3:** Exploração e análise dos dados
- [x] **Etapa 4:** Treinamento do modelo Decision Tree Classifier
- [x] **Etapa 5:** Avaliação do modelo com métricas de desempenho
- [x] **Etapa 6:** Visualização da Matriz de Confusão
- [x] **Etapa 7:** Interface interativa para classificação de novas flores

---

## 📈 Resultados

### Métricas de Desempenho

**Acurácia do modelo:** _[Será preenchida após execução]_

### Relatório de Classificação

```
               precision    recall  f1-score   support

      Setosa       X.XX      X.XX      X.XX        XX
  Versicolor       X.XX      X.XX      X.XX        XX
   Virginica       X.XX      X.XX      X.XX        XX

    accuracy                           X.XX        XX
   macro avg       X.XX      X.XX      X.XX        XX
weighted avg       X.XX      X.XX      X.XX        XX
```

_Os valores serão preenchidos automaticamente após a execução do pipeline._

---

## 📂 Arquivos Gerados

Após a execução do pipeline, os seguintes arquivos são criados automaticamente:

| Arquivo | Descrição | Tamanho aproximado |
|---------|-----------|-------------------|
| `prepared_data.pkl` | Dados preprocessados (treino e teste) salvos em formato pickle | ~10 KB |
| `trained_model.pkl` | Modelo Decision Tree treinado salvo em formato pickle | ~5 KB |
| `matriz_confusao.png` | Visualização gráfica da matriz de confusão | ~50 KB |

### Formato dos arquivos `.pkl`

Os arquivos pickle contêm estruturas Python serializadas:

**`prepared_data.pkl`:**
```python
{
    'X_train': DataFrame,
    'X_test': DataFrame,
    'y_train': Series,
    'y_test': Series,
    'feature_names': list,
    'species_map': dict
}
```

**`trained_model.pkl`:**
```python
{
    'modelo': DecisionTreeClassifier,
    'y_pred': ndarray
}
```

---

## 📚 Referências

### Dataset
- **UCI Machine Learning Repository - Iris Dataset**
  [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris)

- Fisher, R.A. (1936). *"The use of multiple measurements in taxonomic problems"*
  Annual Eugenics, 7, Part II, 179-188.

### Bibliotecas e Ferramentas

- **scikit-learn:** Machine Learning library
  [https://scikit-learn.org/](https://scikit-learn.org/)

- **pandas:** Data manipulation and analysis
  [https://pandas.pydata.org/](https://pandas.pydata.org/)

- **NumPy:** Numerical computing
  [https://numpy.org/](https://numpy.org/)

- **Matplotlib:** Visualization library
  [https://matplotlib.org/](https://matplotlib.org/)

- **Seaborn:** Statistical data visualization
  [https://seaborn.pydata.org/](https://seaborn.pydata.org/)

### Documentação Adicional

- **Decision Trees - scikit-learn**
  [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)

- **Classification Metrics - scikit-learn**
  [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais como parte do curso de Inteligência Artificial - AG2.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFeature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abrir um Pull Request

---

**Desenvolvido com dedicação pela equipe AG2** 🚀
