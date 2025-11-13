import pickle
import numpy as np

def load_classifier():
    """
    Carrega o modelo treinado e o mapa de espécies.
    """
    try:
        with open('trained_model.pkl', 'rb') as f:
            dados_modelo = pickle.load(f) 
            model = dados_modelo['modelo'] 
            
        with open('prepared_data.pkl', 'rb') as f:
            data = pickle.load(f)
            
        species_map = data.get('species_map')
        
        if species_map is None:
            raise ValueError("'species_map' não encontrado em 'prepared_data.pkl'")
            
        return model, species_map
        
    except FileNotFoundError:
        print("Erro: Não foi possível encontrar 'trained_model.pkl' ou 'prepared_data.pkl'.")
        print("Certifique-se de que os arquivos existem no diretório.")
        return None, None
    except KeyError:
        print("Erro: O arquivo 'trained_model.pkl' não contém a chave 'modelo'.")
        print("Por favor, execute o script de treinamento novamente.")
        return None, None

def get_flower_measurements():
    """
    Solicita ao usuário as 4 medidas da flor e trata erros.
    """
    print("\nPor favor, insira as medidas da flor (em cm):")
    
    try:
        sepal_length = float(input("  Comprimento da Sépala: "))
        sepal_width = float(input("  Largura da Sépala:    "))
        petal_length = float(input("  Comprimento da Pétala: "))
        petal_width = float(input("  Largura da Pétala:     "))
        
        measurements = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        return measurements
        
    except ValueError:
        print("\nErro: Valor inválido. Por favor, insira apenas números (ex: 5.1 ou 3.5).")
        return None
    except EOFError:
        return None

def classify_flower(modelo, measurements, species_map):
    """
    Prevê a espécie da flor e exibe o resultado formatado.
    """
   
    prediction_index = modelo.predict(measurements)[0]
    
    predicted_species_full = species_map.get(prediction_index, "Espécie Desconhecida")
    
    predicted_species_name = predicted_species_full.split('-')[-1].lower()

    print(f"\n--- Resultado da Classificação ---")
    print(f"A espécie prevista é: {predicted_species_name.capitalize()}")
    print("----------------------------------")

    if hasattr(modelo, 'predict_proba'):
        probabilities = modelo.predict_proba(measurements)[0]
        
        print("Probabilidades (Confiança):")
        
        try:
            class_names = [species_map[i].split('-')[-1] for i in sorted(species_map.keys())]
            
            if len(probabilities) == len(class_names):
                for i, name in enumerate(class_names):
                    print(f"  {name.capitalize():<10}: {probabilities[i]:.2%}")
            else:
                 print("  (Não foi possível mapear probabilidades aos nomes das classes.)")
        except Exception:
             print("  (Erro ao exibir probabilidades detalhadas.)")
        print("----------------------------------")


def show_examples():
    """
    Mostra exemplos de entrada para o usuário testar.
    """
    print("----------------------------------------------------")
    print("Exemplos de teste (Comprimento Sépala, Largura Sépala, Comprimento Pétala, Largura Pétala):")
    print("  1. Setosa:     5.1, 3.5, 1.4, 0.2")
    print("  2. Versicolor: 6.0, 2.7, 5.1, 1.6")
    print("  3. Virginica:  7.2, 3.0, 5.8, 1.6")
    print("----------------------------------------------------")

def run_classifier():
    """
    Função principal que executa o loop da interface.
    """
    print("Carregando classificador de Íris 'David'...")
    model, s_map = load_classifier()
    
    if model is None or s_map is None:
        print("Encerrando programa devido a erro no carregamento.")
        return

    print("Classificador carregado com sucesso!")
    show_examples()
    
    while True:
        measurements = get_flower_measurements()
        
        if measurements is not None:
            classify_flower(model, measurements, s_map)
        else:
            pass 
        
        try:
            continue_prompt = input("\nDeseja classificar outra flor? (s/n): ").strip().lower()
            if continue_prompt != 's':
                break
        except EOFError:
            break 
            
    print("\nObrigado por usar o classificador 'David'. Encerrando...")

if __name__ == "__main__":
    run_classifier()