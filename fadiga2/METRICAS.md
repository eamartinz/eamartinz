# 📊 Métricas e Gráficos da CNN de Detecção de Olhos

## Resumo do que foi implementado

O código `src/rascunho.py` agora inclui um pipeline completo de treinamento com geração automática de métricas e visualizações.

## 📁 Arquivos Gerados

Após executar o código, os seguintes arquivos serão salvos na pasta `models/`:

### Gráficos
1. **loss_plot.png** - Gráfico comparativo de Loss (Treino vs Validação)
2. **accuracy_plot.png** - Gráfico comparativo de Acurácia (Treino vs Validação)
3. **confusion_matrix.png** - Matriz de confusão do conjunto de teste
4. **training_metrics_combined.png** - Gráficos combinados (Loss + Acurácia)

### Modelos
1. **eyes_model_final.pth** - Pesos do modelo final (arquivo leve)
2. **eyes_model_complete.pth** - Modelo completo com arquitetura
3. **eyes_model_checkpoint.pth** - Checkpoint com metadados de treinamento
4. **eyes_model_best.pth** - Melhor modelo durante o treinamento (early stopping)

## 🎯 Métricas Monitoradas

### Durante o Treinamento
- **Loss**: Função de perda (CrossEntropyLoss)
  - Acompanhado em treino e validação
  - Ajuda a detectar overfitting/underfitting

- **Acurácia**: Percentual de predições corretas
  - Métricas de treino e validação
  - Identificação de melhor modelo

### Após Treinamento
- **Matriz de Confusão**: 
  - Verdadeiros Positivos (TP) e Verdadeiros Negativos (TN)
  - Falsos Positivos (FP) e Falsos Negativos (FN)
  
- **Relatório de Classificação**:
  - Precision: Taxa de acerto das predições positivas
  - Recall: Taxa de detecção de casos positivos
  - F1-Score: Média harmônica entre Precision e Recall

## 🔧 Funcionalidades Implementadas

### 1. **Loop de Treinamento**
```python
- Treina por N épocas (configurável com NUM_EPOCHS)
- Valida após cada época
- Early stopping: para treinamento se não houver melhora
- Salva o melhor modelo automaticamente
```

### 2. **Avaliação em Teste**
```python
- Avalia desempenho final no conjunto de teste
- Calcula métricas de classificação
- Gera matriz de confusão
```

### 3. **Visualização de Resultados**
```python
- Gráficos de alta qualidade (300 DPI)
- Estilo profissional com Seaborn
- Fácil interpretação de resultados
```

### 4. **Salvamento de Modelos**
```python
- Modelo final treinado
- Checkpoint com otimizador (para retomar treinamento)
- Melhor modelo via early stopping
```

## 📈 Interpretando os Gráficos

### Loss Plot
- **Linha vermelha (Validação)**: Deve diminuir no início e estabilizar
- **Linha azul (Treino)**: Geralmente mais baixa que validação
- **Cruzamento**: Se treino < validação, pode indicar overfitting

### Accuracy Plot
- **Linha verde (Treino)**: Deve aumentar ao longo das épocas
- **Linha vermelha (Validação)**: Deve aumentar junto (indicador de generalização)
- **Diferença grande**: Pode indicar overfitting

### Confusion Matrix
- **Diagonal principal**: Predições corretas (mais altas é melhor)
- **Elementos fora da diagonal**: Erros de classificação
- **Células**: Número de amostras em cada categoria

## 🚀 Como Executar

```bash
cd /home/eduardo/Documentos/Git/eamartinz/fadiga2

# Com o ambiente virtual ativo:
python src/rascunho.py

# Ou:
/path/to/venv/bin/python src/rascunho.py
```

## 📊 Configurações Ajustáveis

No topo do arquivo `src/rascunho.py`, você pode ajustar:

```python
BATCH_SIZE = 32           # Tamanho do lote
LEARNING_RATE = 0.001     # Taxa de aprendizado
NUM_EPOCHS = 20           # Número de épocas
TRAIN_SIZE = 0.8          # Proporção treino (80%)
VAL_SIZE = 0.1            # Proporção validação (10%)
TEST_SIZE = 0.1           # Proporção teste (10%)
```

## 💡 Dicas para Melhorar Resultados

1. **Se há overfitting (loss de treino << loss de validação)**:
   - Aumente o dropout (aumentar valor de `p` em `nn.Dropout`)
   - Reduza o número de parâmetros
   - Use mais data augmentation

2. **Se o modelo não converge**:
   - Ajuste a taxa de aprendizado (`LEARNING_RATE`)
   - Aumente o número de épocas (`NUM_EPOCHS`)
   - Normalize melhor os dados de entrada

3. **Para melhor generalização**:
   - Aumente o tamanho do dataset
   - Use mais augmentações de imagem
   - Implemente regularização L2

## 📝 Notas Importantes

- O treinamento cria arquivos em `models/` - certifique-se que a pasta existe
- Early stopping interrompe treino se não houver melhora por 5 épocas
- Os gráficos são salvos com alta resolução (300 DPI) para apresentações
- Checkpoint permite retomar treinamento de onde parou

---

**Última atualização**: 9 de dezembro de 2025
