# -*- coding: utf-8 -*-
"""
Main Pipeline - Classificador Iris
Integra todos os modulos do projeto em um pipeline completo
"""

import sys
import os
from datetime import datetime

# Importar modulos do projeto
import data_loader
import model_trainer
import model_evaluator
import classifier_interface


def print_banner():
    """
    Exibe o banner inicial com titulo, equipe e informacoes do modelo
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "CLASSIFICADOR IRIS - DECISION TREE")
    print("=" * 70)
    print("\nEquipe:")
    print("  - Alexandre Tommasi")
    print("  - Davi Augusto")
    print("\nModelo: Decision Tree Classifier")
    print("Dataset: Iris (150 amostras, 3 classes)")
    print("Objetivo: Classificar especies de flores Iris")
    print("\nData de execucao:", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    print("=" * 70 + "\n")


def check_dataset_exists():
    """
    Verifica se o arquivo iris.csv existe no diretorio atual

    Returns:
        bool: True se o arquivo existe, False caso contrario
    """
    if not os.path.exists('iris.csv'):
        print("\n" + "!" * 70)
        print("ERRO CRITICO: Arquivo 'iris.csv' nao encontrado!")
        print("!" * 70)
        print("\nO arquivo 'iris.csv' e necessario para executar o pipeline.")
        print("Por favor, certifique-se de que o arquivo esta no diretorio atual.")
        print("\nDiretorio atual:", os.getcwd())
        print("\nEncerrando programa...")
        print("!" * 70 + "\n")
        return False
    return True


def execute_data_loading():
    """
    Executa todas as funcoes do modulo data_loader

    Returns:
        tuple: (X_train, X_test, y_train, y_test) ou None em caso de erro
    """
    try:
        print("\n" + "#" * 70)
        print("FASE 1: CARREGAMENTO E PREPARACAO DOS DADOS")
        print("#" * 70 + "\n")

        # Carregar dados do CSV
        df = data_loader.load_iris_from_csv('iris.csv')

        # Converter especies para inteiros
        df = data_loader.convert_species_to_int(df)

        # Explorar dados
        data_loader.explore_data(df)

        # Dividir dados em treino e teste
        X_train, X_test, y_train, y_test = data_loader.split_data(df)

        # Salvar dados preparados
        data_loader.save_data(X_train, X_test, y_train, y_test)

        print("\n" + "#" * 70)
        print("FASE 1 CONCLUIDA COM SUCESSO!")
        print("#" * 70 + "\n")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"\nERRO na Fase 1 (Carregamento de Dados): {e}")
        raise


def execute_model_training():
    """
    Executa todas as funcoes do modulo model_trainer

    Returns:
        tuple: (modelo, y_pred, accuracy) ou None em caso de erro
    """
    try:
        print("\n" + "#" * 70)
        print("FASE 2: TREINAMENTO DO MODELO")
        print("#" * 70 + "\n")

        # Carregar dados preparados
        X_train, X_test, y_train, y_test, feature_names, species_map = model_trainer.load_prepared_data()

        # Treinar Decision Tree
        modelo = model_trainer.train_decision_tree(X_train, y_train)

        # Avaliar modelo preliminarmente
        y_pred = model_trainer.evaluate_preliminary(modelo, X_test, y_test, species_map)

        # Salvar modelo
        model_trainer.save_model(modelo, y_pred)

        print("\n" + "#" * 70)
        print("FASE 2 CONCLUIDA COM SUCESSO!")
        print("#" * 70 + "\n")

        # Calcular acuracia para retornar
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)

        return modelo, y_pred, accuracy

    except Exception as e:
        print(f"\nERRO na Fase 2 (Treinamento do Modelo): {e}")
        raise


def execute_model_evaluation():
    """
    Executa todas as funcoes do modulo model_evaluator

    Returns:
        float: Acuracia final do modelo ou None em caso de erro
    """
    try:
        print("\n" + "#" * 70)
        print("FASE 3: AVALIACAO DO MODELO")
        print("#" * 70 + "\n")

        # Carregar modelo e dados
        X_test, y_test, model, species_map = model_evaluator.load_model_and_data()

        # Avaliar modelo
        y_pred, accuracy = model_evaluator.evaluate_model(model, X_test, y_test)

        # Criar matriz de confusao
        model_evaluator.create_confusion_matrix(y_test, y_pred)

        print("\n" + "#" * 70)
        print("FASE 3 CONCLUIDA COM SUCESSO!")
        print("#" * 70 + "\n")

        return accuracy

    except Exception as e:
        print(f"\nERRO na Fase 3 (Avaliacao do Modelo): {e}")
        raise


def ask_interactive_mode():
    """
    Pergunta ao usuario se deseja usar a interface interativa

    Returns:
        bool: True se o usuario quer usar a interface, False caso contrario
    """
    print("\n" + "=" * 70)
    print("INTERFACE INTERATIVA")
    print("=" * 70)
    print("\nDeseja usar a interface interativa para classificar flores?")
    print("Voce podera inserir medidas de flores e ver as predicoes em tempo real.")

    while True:
        try:
            resposta = input("\nUsar interface interativa? (s/n): ").strip().lower()
            if resposta in ['s', 'sim', 'y', 'yes']:
                return True
            elif resposta in ['n', 'nao', 'no']:
                return False
            else:
                print("Resposta invalida. Por favor, digite 's' para sim ou 'n' para nao.")
        except EOFError:
            return False


def print_final_summary(accuracy):
    """
    Exibe o resumo final da execucao do pipeline

    Args:
        accuracy: Acuracia final do modelo
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "RESUMO FINAL DA EXECUCAO")
    print("=" * 70)

    print("\n--- Metricas do Modelo ---")
    print(f"Acuracia Final: {accuracy * 100:.2f}%")

    print("\n--- Arquivos Gerados ---")
    arquivos = [
        ('prepared_data.pkl', 'Dados preprocessados (treino e teste)'),
        ('trained_model.pkl', 'Modelo Decision Tree treinado'),
        ('matriz_confusao.png', 'Visualizacao da matriz de confusao')
    ]

    for arquivo, descricao in arquivos:
        if os.path.exists(arquivo):
            tamanho = os.path.getsize(arquivo)
            tamanho_kb = tamanho / 1024
            print(f"  {arquivo:<25} - {descricao} ({tamanho_kb:.2f} KB)")
        else:
            print(f"  {arquivo:<25} - {descricao} (NAO ENCONTRADO)")

    print("\n--- Estatisticas do Pipeline ---")
    print(f"  Total de fases executadas: 3")
    print(f"  Status: SUCESSO")
    print(f"  Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    print("\n" + "=" * 70)
    print(" " * 15 + "PIPELINE CONCLUIDO COM SUCESSO!")
    print("=" * 70 + "\n")


def main():
    """
    Funcao principal que orquestra todo o pipeline
    """
    # Exibir banner inicial
    print_banner()

    # Verificar se o dataset existe
    if not check_dataset_exists():
        sys.exit(1)

    # Fase 1: Carregamento e preparacao dos dados
    X_train, X_test, y_train, y_test = execute_data_loading()

    # Fase 2: Treinamento do modelo
    modelo, y_pred, accuracy_train = execute_model_training()

    # Fase 3: Avaliacao do modelo
    accuracy_final = execute_model_evaluation()

    # Perguntar sobre interface interativa
    if ask_interactive_mode():
        print("\n" + "=" * 70)
        print("INICIANDO INTERFACE INTERATIVA")
        print("=" * 70 + "\n")

        # Executar interface interativa
        classifier_interface.run_classifier()

        print("\n" + "=" * 70)
        print("INTERFACE INTERATIVA ENCERRADA")
        print("=" * 70)
    else:
        print("\nInterface interativa nao sera executada.")

    # Exibir resumo final
    print_final_summary(accuracy_final)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "!" * 70)
        print("EXECUCAO INTERROMPIDA PELO USUARIO (Ctrl+C)")
        print("!" * 70 + "\n")
        sys.exit(0)
    except Exception as e:
        print("\n\n" + "!" * 70)
        print("ERRO FATAL DURANTE A EXECUCAO DO PIPELINE")
        print("!" * 70)
        print(f"\nDetalhes do erro: {e}")
        print(f"Tipo do erro: {type(e).__name__}")
        print("\nPor favor, verifique os logs acima para mais informacoes.")
        print("\n" + "!" * 70 + "\n")
        sys.exit(1)
