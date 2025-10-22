# NetTraining
Bibilioteca feita para treinamento de redes neurais de forma facilitada (Uso pessoal principalmente)

# Métodos

#### `shard_data(data: torch.Tensor) -> List[torch.Tensor]`
Divide um tensor em fatias (shards)

**Args:**
- `data (torch.Tensor)`: Tensor de entrada.

**Returns:**
- `List[torch.Tensor]`: Lista com shards do tensor.

---

#### `train(x, y, epochs=1000, reps=1, phased=False, plot=True, reportProgress=True, limitToMaxAccuracy=False)`
Executa o loop completo de treinamento, com suporte a checkpoint automático e "phase training", onde ele introduz o modelo a pequenos shards de dados.

**Args:**
- `x (torch.Tensor)`: Features de entrada.  
- `y (torch.Tensor)`: Valores alvo.  
- `epochs (int)`: Número de épocas por repetição (default: 1000).  
- `reps (int)`: Número de repetições (default: 1).  
- `phased (bool)`: Ativa treinamento *shard-based* (default: False).  
- `plot (bool)`: Exibe o gráfico de perda (default: True).  
- `reportProgress (bool)`: Mostra progresso e métricas (default: True).  
- `limitToMaxAccuracy (bool)`: Interrompe quando atingir 100% de acurácia (default: False). -- Não recomendado

---

#### `evaluate(x: torch.Tensor)`
Avalia o modelo e printa previsões 

**Args:**
- `x (torch.Tensor)`: Tensor de entrada para predição.

# Exemplo de uso
```python
import torch
import torch.nn as nn
import torch.optim as optim
from NetTrainer import Trainer  

#  (f(x) = x + 2)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.arange(-10, 10, 0.1).unsqueeze(1)
y = x + 2

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

trainer = Trainer(model, optimizer, criterion, device)

trainer.train(
    x, y,
    epochs=1000,
    phased=False,
    plot=True,
    reportProgress=True
)

trainer.evaluate(torch.tensor([[5.0], [10.0], [20.0]]))
```
