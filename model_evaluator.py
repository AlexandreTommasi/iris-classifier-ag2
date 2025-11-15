import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_model_and_data():
    """
    Carrega os dados de teste e o modelo treinado.
    """
    print("Carregando dados e modelo...")
    with open('prepared_data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    with open('trained_model.pkl', 'rb') as f:
        dados_modelo = pickle.load(f) 
        model = dados_modelo['modelo'] 

    X_test = data.get('X_test')
    y_test = data.get('y_test')
    species_map = data.get('species_map')

    if X_test is None or y_test is None or species_map is None:
        raise ValueError("Arquivo 'prepared_data.pkl' não contém X_test, y_test ou species_map.")

    return X_test, y_test, model, species_map

def evaluate_model(modelo, X_test, y_test):
    """
    Calcula a acurácia e o relatório de classificação.
    """
    print("Avaliando modelo...")
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    labels_list = [1, 2, 3]
    target_names_list = ['Setosa', 'Versicolor', 'Virginica']
    
    report = classification_report(
        y_test, 
        y_pred, 
        labels=labels_list, 
        target_names=target_names_list
    )
    
    print("\n--- Relatório de Classificação ---")
    print(report)
    print("----------------------------------")
    
    return y_pred, accuracy

def create_confusion_matrix(y_test, y_pred):
    """
    Cria, salva e interpreta a matriz de confusão.
    """
    print("Gerando Matriz de Confusão...")
    labels_list = [1, 2, 3]
    display_labels = ['Setosa', 'Versicolor', 'Virginica']
    
    cm = confusion_matrix(y_test, y_pred, labels=labels_list)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm, 
        annot=False, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=display_labels, 
        yticklabels=display_labels,
        cbar=True 
    )
    
    plt.title('Matriz de Confusão', fontsize=18)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Prevista', fontsize=12)

    thresh = cm.max() / 2.
    

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            
            ax.text(
                j + 0.5,
                i + 0.5, 
                f"{cm[i, j]}", 
                ha="center", 
                va="center",
                
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16 # Tamanho da fonte
            )
    
    plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
    print(f"Matriz de confusão salva como 'matriz_confusao.png'")

    print("\n--- Interpretação da Matriz ---")
    print("A diagonal principal (de cima-esquerda para baixo-direita) mostra os acertos.")
    print("Valores fora da diagonal indicam classificações incorretas (erros).")
    print("-------------------------------")

if __name__ == "__main__":
    try:
        X_test, y_test, model, _ = load_model_and_data()
        y_pred, final_accuracy = evaluate_model(model, X_test, y_test)
        create_confusion_matrix(y_test, y_pred)
        
        print(f"\n✅ Acurácia Final do Modelo: {final_accuracy:.4f}")
        
    except FileNotFoundError:
        print("Erro: Arquivos 'prepared_data.pkl' ou 'trained_model.pkl' não encontrados.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")