<style>
    h2 {
        font-size: 24px;
    }
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# 1.6 - Exercícios resolvidos

Esta seção contém a resolução dos exercícios propostos no Capítulo 2 do livro texto de referência.

## Exercício 1 - Modelo de máxima verossimilhança
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Configuração da semente para manter sempre os mesmos resultados
np.random.seed(123456)

# Caminho do arquivo de saída
path_out = 'saidaSim.txt'

# Parâmetros das distribuições
mu1 = [0.0, 0.5]
Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
rotulo1 = 1
qnt1 = 20

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
rotulo2 = 1
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
rotulo3 = 2
qnt3 = 50

conjMu = np.array([mu1, mu2, mu3])
conjSigma = np.array([Sigma1, Sigma2, Sigma3])
rotulos = np.array([rotulo1, rotulo2, rotulo3])
quantidades = np.array([qnt1, qnt2, qnt3])

# Criar e gravar dados
f = open(path_out, 'w')

# Listas para armazenar dados
todos_dados = []
todos_rotulos = []

for r in range(rotulos.size):
    dados = np.random.multivariate_normal(conjMu[r, :], conjSigma[r, :, :], quantidades[r])

    # Gravar dados e armazenar nos arrays
    for i in range(quantidades[r]):
        st = str(rotulos[r])
        st += ',' + ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n'
        f.write(st)

    # Armazenar dados e rótulos
    todos_dados.append(dados)
    todos_rotulos.extend([rotulos[r]] * quantidades[r])  # Adiciona o rótulo correspondente

f.close()

# Concatenar todos os dados e rótulos em arrays numpy
todos_dados = np.concatenate(todos_dados)  # Todos os dados concatenados
todos_rotulos = np.array(todos_rotulos)  # Rótulos em formato array

# Criar e treinar o modelo Naive Bayes
modelo = GaussianNB()
modelo.fit(todos_dados, todos_rotulos)

# Contar o total por classe
contagem_por_classe = {rotulo: np.sum(todos_rotulos == rotulo) for rotulo in np.unique(todos_rotulos)}

# Exibir a contagem total por classe
print("Contagem total por rótulo:")
for rotulo, contagem in contagem_por_classe.items():
    print(f'Rótulo {rotulo}: {contagem}')

# Plotar os dados
plt.figure(figsize=(10, 6))

# Plotar cada classe em cores diferentes
plt.scatter(todos_dados[todos_rotulos == 1][:, 0], todos_dados[todos_rotulos == 1][:, 1], label='Classe 1', alpha=0.5, color='blue')
plt.scatter(todos_dados[todos_rotulos == 2][:, 0], todos_dados[todos_rotulos == 2][:, 1], label='Classe 2', alpha=0.5, color='orange')

plt.title('Distribuições das Classes')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.grid()
plt.show()
```

### Resultados
Contagem total por classe (rótulo):

* Rótulo 1: 60
* Rótulo 2: 50

<div align="center"> 
    
![figura1](../images/ml.png "figura 1") <legend>Figura 1.4 - Resultado ao executar código do exercício 1. Fonte: Elaborado pelo autor.
</legend>
</div>

## Exercício 2 - Modelo Naive Bayes
```
import numpy as np
import matplotlib.pyplot as plt

# Configuração da semente para manter sempre os mesmos resultados
np.random.seed(123456)

# Caminho do arquivo de saída
path_out = 'saidaSim.txt'

# Parâmetros das distribuições
mu1 = [0.0, 0.5]
Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
rotulo1 = 1
qnt1 = 20

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
rotulo2 = 1
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
rotulo3 = 2
qnt3 = 50

conjMu = np.array([mu1, mu2, mu3])
conjSigma = np.array([Sigma1, Sigma2, Sigma3])
rotulos = np.array([rotulo1, rotulo2, rotulo3])
quantidades = np.array([qnt1, qnt2, qnt3])

# Criar e gravar dados
f = open(path_out, 'w')

# Listas para armazenar dados
todos_dados = []
todos_rotulos = []

for r in range(rotulos.size):
    dados = np.random.multivariate_normal(conjMu[r, :], conjSigma[r, :, :], quantidades[r])

    # Gravar dados e armazenar nos arrays
    for i in range(quantidades[r]):
        st = str(rotulos[r])
        st += ',' + ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n'
        f.write(st)

    # Armazenar dados e rótulos
    todos_dados.append(dados)
    todos_rotulos.extend([rotulos[r]] * quantidades[r])  # Adiciona o rótulo correspondente

f.close()

# Concatenar todos os dados e rótulos em arrays numpy
todos_dados = np.concatenate(todos_dados)  # Todos os dados concatenados
todos_rotulos = np.array(todos_rotulos)  # Rótulos em formato array

# Função para calcular a probabilidade de uma classe usando a distribuição normal
def calcular_probabilidade(x, mu, sigma):
    d = mu.shape[0]
    coef = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
    expoente = -0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)
    return coef * np.exp(expoente)

# Função para prever a classe usando Naive Bayes
def prever(todos_dados, conjMu, conjSigma, rotulos):
    previsoes = []
    for x in todos_dados:
        probabilidades = []
        for i in range(len(rotulos)):
            prob = calcular_probabilidade(x, conjMu[i], conjSigma[i])
            probabilidades.append(prob)
        # O rótulo da classe com a maior probabilidade é a previsão
        previsoes.append(rotulos[np.argmax(probabilidades)])
    return np.array(previsoes)

# Fazer previsões
previsoes = prever(todos_dados, conjMu, conjSigma, rotulos)

# Calcular a acurácia
acuracia = np.mean(previsoes == todos_rotulos)
print(f'Acurácia do modelo Naive Bayes: {acuracia:.2f}')

# Contar o total por classe
contagem_por_classe = {rotulo: np.sum(todos_rotulos == rotulo) for rotulo in np.unique(todos_rotulos)}

# Exibir a contagem total por classe
print("Contagem total por classe (rótulo):")
for rotulo, contagem in contagem_por_classe.items():
    print(f'Rótulo {rotulo}: {contagem}')

# Plotar os dados
plt.figure(figsize=(10, 6))

# Plotar cada classe em cores diferentes
plt.scatter(todos_dados[todos_rotulos == 1][:, 0], todos_dados[todos_rotulos == 1][:, 1], label='Classe 1', alpha=0.5, color='blue')
plt.scatter(todos_dados[todos_rotulos == 2][:, 0], todos_dados[todos_rotulos == 2][:, 1], label='Classe 2', alpha=0.5, color='orange')

plt.title('Distribuições das Classes')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.grid()
plt.show()

```
### Resultados

Acurácia do modelo Naive Bayes: 0.95

Contagem total por classe (rótulo):
* Rótulo 1: 60
* Rótulo 2: 50

<div align="center"> 
    
![figura1](../images/nb.png "figura 1") <legend>Figura 1.5 - Resultado ao executar código do exercício 2. Fonte: Elaborado pelo autor.
</legend>
</div>


## Exercício 3 - Classificador KNN
```
import numpy as np
import matplotlib.pyplot as plt

# Configuração da semente para reprodutibilidade
np.random.seed(123456)

# Caminho do arquivo de saída
path_out = 'saidaSim.txt'

# Parâmetros das distribuições
mu1 = [0.0, 0.5]; Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
rotulo1 = 1; qnt1 = 20

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
rotulo2 = 1
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
rotulo3 = 2
qnt3 = 50

conjMu = np.array([mu1, mu2, mu3])
conjSigma = np.array([Sigma1, Sigma2, Sigma3])
rotulos = np.array([rotulo1, rotulo2, rotulo3])
quantidades = np.array([qnt1, qnt2, qnt3])

# Criar e gravar dados
f = open(path_out, 'w')

# Listas para armazenar dados
todos_dados = []
todos_rotulos = []

for r in range(rotulos.size):
    dados = np.random.multivariate_normal(conjMu[r, :], conjSigma[r, :, :], quantidades[r])

    # Gravar dados e armazenar nos arrays
    for i in range(quantidades[r]):
        st = str(rotulos[r])
        st += ',' + ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n'
        f.write(st)

    # Armazenar dados e rótulos
    todos_dados.append(dados)
    todos_rotulos.extend([rotulos[r]] * quantidades[r])  # Adiciona o rótulo correspondente

f.close()

# Concatenar todos os dados e rótulos em arrays numpy
todos_dados = np.concatenate(todos_dados)  # Todos os dados concatenados
todos_rotulos = np.array(todos_rotulos)  # Rótulos em formato array

# Função para calcular a distância euclidiana
def calcular_distancia(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Função KNN
def knn(todos_dados, todos_rotulos, ponto, k):
    distancias = []

    # Calcular distâncias entre o ponto e todos os dados
    for i in range(len(todos_dados)):
        d = calcular_distancia(ponto, todos_dados[i])
        distancias.append((d, todos_rotulos[i]))

    # Ordenar as distâncias e pegar os k vizinhos mais próximos
    distancias.sort(key=lambda x: x[0])
    vizinhos_k = distancias[:k]

    # Contar os rótulos dos vizinhos
    contagem_rotulos = {}
    for _, rotulo in vizinhos_k:
        if rotulo in contagem_rotulos:
            contagem_rotulos[rotulo] += 1
        else:
            contagem_rotulos[rotulo] = 1

    # Retornar o rótulo mais frequente
    return max(contagem_rotulos, key=contagem_rotulos.get)

# Função para calcular a porcentagem de acertos
def calcular_acertos(todos_dados, todos_rotulos, k):
    acertos = 0
    for i in range(len(todos_dados)):
        ponto = todos_dados[i]
        rotulo_predito = knn(todos_dados, todos_rotulos, ponto, k)
        if rotulo_predito == todos_rotulos[i]:
            acertos += 1
    return acertos / len(todos_rotulos) * 100  # Porcentagem de acertos

# Calcular acertos para diferentes valores de k
valores_k = [3, 11, 25]
for k in valores_k:
    acertos = calcular_acertos(todos_dados, todos_rotulos, k)
    print(f'Porcentagem de acertos com k={k}: {acertos:.2f}%')

# Plotar os dados
plt.figure(figsize=(10, 6))

# Plotar cada classe em cores diferentes
plt.scatter(todos_dados[todos_rotulos == 1][:, 0], todos_dados[todos_rotulos == 1][:, 1], label='Classe 1', alpha=0.5, color='blue')
plt.scatter(todos_dados[todos_rotulos == 2][:, 0], todos_dados[todos_rotulos == 2][:, 1], label='Classe 2', alpha=0.5, color='orange')

plt.title('Distribuições das Classes')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.grid()
plt.show()
```

### Resultados

* Porcentagem de acertos com k=3: 94.55%
* Porcentagem de acertos com k=11: 93.64%
* Porcentagem de acertos com k=25: 92.73%

<div align="center"> 
    
![figura1](../images/nb.png "figura 1") <legend>Figura 1.6 - Resultado ao executar código do exercício 3. Fonte: Elaborado pelo autor.
</legend>
</div>
