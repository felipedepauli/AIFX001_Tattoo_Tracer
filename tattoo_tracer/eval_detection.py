import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tattoo_tracer import TattooTracer  # Assumindo que TattooTracer está no arquivo tattoo_tracer.py


class TattooTracerEvaluator:
    def __init__(self, tattoo_dir, non_tattoo_dir, model_path):
        """
        Inicializa o avaliador com diretórios para imagens com e sem tatuagens.
        """
        self.tattoo_dir = tattoo_dir
        self.non_tattoo_dir = non_tattoo_dir
        self.tracer = TattooTracer()
        # Atualize o caminho do modelo, se necessário
        # self.tracer.tattoo_detector.model = self.tracer.tattoo_detector.load_model(model_path)
        
    def evaluate(self):
        """
        Avalia o desempenho do tattoo_tracer em classificar imagens com e sem tatuagens.
        """
        labels = []
        predictions = []
        tattoo_confidences = []  # Armazena as confianças para tatuagens
        non_tattoo_confidences = []  # Armazena as confianças para não tatuagens
        
        # Processa imagens com tatuagens
        for image_file in os.listdir(self.tattoo_dir):
            img_path = os.path.join(self.tattoo_dir, image_file)
            img = cv2.imread(img_path)
            prediction, confidence = self.tracer.tattoo_detector.detect_tattoo(img)
            labels.append(1)  # 1 para tatuagem
            predictions.append(1 if confidence > 0.5 else 0)
            tattoo_confidences.append(confidence)

        # Processa imagens sem tatuagens
        for image_file in os.listdir(self.non_tattoo_dir):
            img_path = os.path.join(self.non_tattoo_dir, image_file)
            img = cv2.imread(img_path)
            prediction, confidence = self.tracer.tattoo_detector.detect_tattoo(img)
            labels.append(0)  # 0 para não tatuagem
            predictions.append(1 if confidence > 0.5 else 0)
            non_tattoo_confidences.append(confidence)

        # Calcula e imprime as confianças médias
        tattoo_confidence_mean = np.mean(tattoo_confidences)
        non_tattoo_confidence_mean = np.mean(non_tattoo_confidences)
        print(f'Média da confiança para imagens com tatuagem         : {tattoo_confidence_mean:.4f}')
        print(f'Média da confiança para imagens sem tatuagem         : {non_tattoo_confidence_mean:.4f}')
        print(f'Desvio padrão da confiança para imagens com tatuagem : {np.std(tattoo_confidences):.4f}')
        print(f'Mediana da confiança para imagens com tatuagem       : {np.median(tattoo_confidences):.4f}')
        print(f'Desvio padrão da confiança para imagens sem tatuagem : {np.std(non_tattoo_confidences):.4f}')
        print(f'Mediana da confiança para imagens sem tatuagem       : {np.median(non_tattoo_confidences):.4f}')

        # Gera relatório de classificação e matriz de confusão
        print(classification_report(labels, predictions, target_names=['Non-Tattoo', 'Tattoo']))
        print(confusion_matrix(labels, predictions))

if __name__ == "__main__":
    # Definir diretórios e caminho do modelo conforme necessário
    evaluator = TattooTracerEvaluator(tattoo_dir='../ml_pipeline/ml_02_data_ready/data/tattoo_trace/detection/tattoo', non_tattoo_dir='../ml_pipeline/ml_02_data_ready/data/tattoo_trace/detection/non_tattoo', model_path='./last_doideira.h5')
    evaluator.evaluate()
