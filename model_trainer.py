# -*- coding: utf-8 -*-
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_prepared_data():
    """
    Carrega os dados preparados do arquivo pickle

    Returns:
        X_train, X_test, y_train, y_test, feature_names, species_map
    """
    print("=" * 70)
    print("CARREGANDO DADOS PREPARADOS")
    print("=" * 70)

    filename = 'prepared_data.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']
    species_map = data['species_map']

    print(f"\nArquivo '{filename}' carregado com sucesso!")
    print("\nDados carregados:")
    print(f"  - X_train: {X_train.shape} (treino)")
    print(f"  - X_test: {X_test.shape} (teste)")
    print(f"  - y_train: {y_train.shape} (treino)")
    print(f"  - y_test: {y_test.shape} (teste)")
    print(f"  - Features: {feature_names}")
    print(f"  - Mapa de especies: {species_map}")

    print("\n--- Distribuicao das classes no treino ---")
    for cls in sorted(y_train.unique()):
        count = (y_train == cls).sum()
        print(f"  Classe {cls} ({species_map[cls]}): {count} amostras")

    return X_train, X_test, y_train, y_test, feature_names, species_map


def train_decision_tree(X_train, y_train):
    """
    Treina um modelo Decision Tree Classifier

    Args:
        X_train: Features de treino
        y_train: Labels de treino

    Returns:
        Modelo treinado
    """
    print("\n" + "=" * 70)
    print("ETAPA 4: TREINANDO MODELO DECISION TREE")
    print("=" * 70)

    print("\nCriando Decision Tree Classifier...")
    print("Parametros: random_state=42")

    modelo = DecisionTreeClassifier(random_state=42)

    print("\nTreinando modelo...")
    modelo.fit(X_train, y_train)

    print("\nModelo treinado com sucesso!")
    print(f"Profundidade da arvore: {modelo.get_depth()}")
    print(f"Numero de folhas: {modelo.get_n_leaves()}")
    print(f"Numero de features: {modelo.n_features_in_}")

    return modelo


def evaluate_preliminary(modelo, X_test, y_test, species_map):
    """
    Avalia o modelo preliminarmente

    Args:
        modelo: Modelo treinado
        X_test: Features de teste
        y_test: Labels de teste
        species_map: Dicionario de mapeamento das especies

    Returns:
        y_pred: Previsoes do modelo
    """
    print("\n" + "=" * 70)
    print("AVALIACAO PRELIMINAR DO MODELO")
    print("=" * 70)

    print("\nFazendo previsoes no conjunto de teste...")
    y_pred = modelo.predict(X_test)

    print("\nCalculando acuracia...")
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAcuracia no conjunto de teste: {accuracy * 100:.2f}%")

    print("\n--- Primeiras 10 previsoes vs valores reais ---")
    print(f"{'ID':<5} {'Previsto':<20} {'Real':<20} {'Correto?':<10}")
    print("-" * 60)

    for i in range(min(10, len(y_test))):
        pred_name = species_map[y_pred[i]]
        real_name = species_map[y_test.iloc[i]]
        correto = "SIM" if y_pred[i] == y_test.iloc[i] else "NAO"
        print(f"{i:<5} {pred_name:<20} {real_name:<20} {correto:<10}")

    # Contar acertos
    acertos = (y_pred == y_test).sum()
    total = len(y_test)
    print("\n" + "-" * 60)
    print(f"Total de acertos: {acertos}/{total}")

    return y_pred


def save_model(modelo, y_pred):
    """
    Salva o modelo treinado e as previsoes em arquivo pickle

    Args:
        modelo: Modelo treinado
        y_pred: Previsoes do modelo
    """
    print("\n" + "=" * 70)
    print("SALVANDO MODELO TREINADO")
    print("=" * 70)

    model_data = {
        'modelo': modelo,
        'y_pred': y_pred
    }

    filename = 'trained_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModelo salvo com sucesso em '{filename}'!")
    print("\nConteudo do arquivo:")
    print("  - modelo: DecisionTreeClassifier treinado")
    print(f"  - y_pred: {y_pred.shape} previsoes")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INICIANDO TREINAMENTO DO MODELO")
    print("=" * 70)

    # Carregar dados preparados
    X_train, X_test, y_train, y_test, feature_names, species_map = load_prepared_data()

    # Treinar Decision Tree
    modelo = train_decision_tree(X_train, y_train)

    # Avaliar modelo preliminarmente
    y_pred = evaluate_preliminary(modelo, X_test, y_test, species_map)

    # Salvar modelo
    save_model(modelo, y_pred)

    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUIDO COM SUCESSO!")
    print("=" * 70)
