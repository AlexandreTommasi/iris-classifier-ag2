# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


def load_iris_from_csv(filename='iris.csv'):
    """
    Etapa 1: Carrega o dataset Iris de um arquivo CSV

    Args:
        filename: Nome do arquivo CSV a ser carregado

    Returns:
        DataFrame com os dados do Iris
    """
    print("=" * 70)
    print("ETAPA 1: CARREGANDO DADOS DO ARQUIVO CSV")
    print("=" * 70)

    df = pd.read_csv(filename)

    print(f"\nArquivo '{filename}' carregado com sucesso!")
    print(f"\nDimensoes do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("\n--- Primeiras 5 linhas do dataset ---")
    print(df.head())

    print("\n--- Tipos de dados ---")
    print(df.dtypes)

    print("\n--- Informacoes gerais ---")
    print(df.info())

    return df


def convert_species_to_int(df):
    """
    Etapa 2: Converte as especies de string para inteiros

    Args:
        df: DataFrame com os dados

    Returns:
        DataFrame com a coluna species convertida para int
    """
    print("\n" + "=" * 70)
    print("ETAPA 2: CONVERTENDO ESPECIES PARA VALORES INTEIROS")
    print("=" * 70)

    print("\n--- Valores ANTES da conversao ---")
    print(f"Tipo da coluna 'species': {df['species'].dtype}")
    print(f"Valores unicos:\n{df['species'].value_counts()}")

    # Criar uma copia para nao modificar o original
    df_converted = df.copy()

    # Conversao das especies para inteiros
    df_converted['species'] = df_converted['species'].replace({
        'Iris-setosa': 1,
        'Iris-versicolor': 2,
        'Iris-virginica': 3
    }).astype('int64')

    print("\n--- Valores DEPOIS da conversao ---")
    print(f"Tipo da coluna 'species': {df_converted['species'].dtype}")
    print(f"Valores unicos:\n{df_converted['species'].value_counts()}")

    print("\nMapeamento aplicado:")
    print("  Iris-setosa     -> 1")
    print("  Iris-versicolor -> 2")
    print("  Iris-virginica  -> 3")

    return df_converted


def explore_data(df):
    """
    Etapa 3: Explora os dados do dataset

    Args:
        df: DataFrame com os dados
    """
    print("\n" + "=" * 70)
    print("ETAPA 3: EXPLORANDO OS DADOS")
    print("=" * 70)

    print("\n--- Informacoes gerais do dataset ---")
    df.info()

    print("\n--- Estatisticas descritivas ---")
    print(df.describe())

    print("\n--- Primeiras linhas ---")
    print(df.head(10))

    print("\n--- Distribuicao das classes ---")
    class_distribution = df['species'].value_counts().sort_index()
    print(class_distribution)

    print("\nDistribuicao percentual:")
    for species, count in class_distribution.items():
        percentage = (count / len(df)) * 100
        species_name = {1: 'Iris-setosa', 2: 'Iris-versicolor', 3: 'Iris-virginica'}[species]
        print(f"  Classe {species} ({species_name}): {count} amostras ({percentage:.1f}%)")

    print(f"\nTotal de amostras: {len(df)}")


def split_data(df):
    """
    Etapa 3 (continuacao): Divide os dados em conjuntos de treino e teste

    Args:
        df: DataFrame com os dados

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("DIVIDINDO DADOS EM TREINO E TESTE")
    print("=" * 70)

    # Separar features (X) e target (y)
    X = df.drop('species', axis=1)
    y = df['species']

    print(f"\nFeatures (X): {X.shape[1]} colunas")
    print(f"Colunas: {list(X.columns)}")
    print(f"\nTarget (y): {y.name}")
    print(f"Classes: {sorted(y.unique())}")

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y
    )

    print("\n--- Divisao dos dados ---")
    print(f"Conjunto de TREINO: {len(X_train)} amostras ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Conjunto de TESTE:  {len(X_test)} amostras ({len(X_test)/len(df)*100:.1f}%)")

    print("\n--- Distribuicao das classes no conjunto de TREINO ---")
    train_distribution = y_train.value_counts().sort_index()
    for species, count in train_distribution.items():
        percentage = (count / len(y_train)) * 100
        print(f"  Classe {species}: {count} amostras ({percentage:.1f}%)")

    print("\n--- Distribuicao das classes no conjunto de TESTE ---")
    test_distribution = y_test.value_counts().sort_index()
    for species, count in test_distribution.items():
        percentage = (count / len(y_test)) * 100
        print(f"  Classe {species}: {count} amostras ({percentage:.1f}%)")

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    """
    Salva os dados processados em arquivo pickle

    Args:
        X_train, X_test, y_train, y_test: Dados divididos
    """
    print("\n" + "=" * 70)
    print("SALVANDO DADOS PROCESSADOS")
    print("=" * 70)

    # Criar dicionario com todos os dados
    feature_names = list(X_train.columns)
    species_map = {
        1: 'Iris-setosa',
        2: 'Iris-versicolor',
        3: 'Iris-virginica'
    }

    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'species_map': species_map
    }

    # Salvar em pickle
    filename = 'prepared_data.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"\nDados salvos com sucesso em '{filename}'!")
    print("\nConteudo do arquivo:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - y_test: {y_test.shape}")
    print(f"  - feature_names: {feature_names}")
    print(f"  - species_map: {species_map}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INICIANDO PROCESSAMENTO DO DATASET IRIS")
    print("=" * 70)

    # Etapa 1: Carregar dados
    df = load_iris_from_csv('iris.csv')

    # Etapa 2: Converter especies para inteiros
    df = convert_species_to_int(df)

    # Etapa 3: Explorar dados
    explore_data(df)

    # Etapa 3 (continuacao): Dividir dados
    X_train, X_test, y_train, y_test = split_data(df)

    # Salvar dados processados
    save_data(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
    print("=" * 70)
