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

## Código de referência para resolver os exercícios.
``` 
for i in range(quantidades[r]):
        st = str(rotulos[r])
        st += ',' + ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n' #Força continuação na próxima linha
        f.write(st)
f.close() #Encerra o processo de escrita dos dados
```

## Exercício 1 - Modelo de máxima verossimilhança

Considere um problema de classificação envolvendo duas classes, $\omega_{1}$ e $\omega_{2}$, para o qual será empregado o modelo ML. Efetue a modelagem das distribuições de probabilidade (gaussianas multivariadas) referetes às classes $\omega_{1}$ e $\omega_{2}$ com base em um conjunto de dados $D$. Posteriormente, realize a classificação de um segundo conjunto de dados $I$ e contabilize o número de previsões corretas proporcionadas com base no rótulo associado a cada exemplo.

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
    
![figura1](../images/ml.png "figura 14") <legend>Figura 1.4 - Resultado ao executar código do exercício 1. Fonte: Elaborado pelo autor.
</legend>
</div>

## Exercício 2 - Modelo Naive Bayes

 Refaça o exercício anterior com uso do modelo Naive Bayes.

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
    
![figura1](../images/nb.png "figura 15") <legend>Figura 1.5 - Resultado ao executar código do exercício 2. Fonte: Elaborado pelo autor.
</legend>
</div>


## Exercício 3 - Classificador KNN
Utilize os conjuntos $D$ e $I$ obtidos no Exercício 1 para modelagem e aplicação do classificador KNN. Verifique a porcentagem de acerto deste método assumindo:
* $k = 3$
* $k = 11$
* $k = 25$
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
    
![figura1](../images/nb.png "figura 16") <legend>Figura 1.6 - Resultado ao executar código do exercício 3. Fonte: Elaborado pelo autor.
</legend>
</div>


## Exercício 4 - Método GMM
Aplique o método GMM, baseado em distribuições gaussianas multivariadas, a fim de modelar o comportamento da distribuição de probabilidade dos dados gerados no Exercício 1.

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Configuração da semente para manter sempre os mesmos resultados
np.random.seed(123456)

# Caminho do arquivo de saída
path_out = 'saidaSim.txt'

# Parâmetros das distribuições
mu1 = [0.0, 0.5]
Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
qnt1 = 20

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
qnt3 = 50

conjMu = np.array([mu1, mu2, mu3])
conjSigma = np.array([Sigma1, Sigma2, Sigma3])
quantidades = np.array([qnt1, qnt2, qnt3])

# Criar e gravar dados
f = open(path_out, 'w')

# Listas para armazenar dados
todos_dados = []

for r in range(conjMu.shape[0]):
    dados = np.random.multivariate_normal(conjMu[r], conjSigma[r], quantidades[r])

    # Gravar dados e armazenar nos arrays
    for i in range(quantidades[r]):
        st = ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n'
        f.write(st)

    # Armazenar dados
    todos_dados.append(dados)

f.close()

# Concatenar todos os dados em arrays numpy
todos_dados = np.concatenate(todos_dados)  # Todos os dados concatenados

# Aplicar o modelo GMM (Gaussian Mixture Model)
num_componentes = 3
modelo_gmm = GaussianMixture(n_components=num_componentes, covariance_type='full', random_state=42)
modelo_gmm.fit(todos_dados)

# Prever as classes com base no GMM
labels = modelo_gmm.predict(todos_dados)

# Plotar os dados e as distribuições modeladas
plt.figure(figsize=(10, 6))
plt.scatter(todos_dados[:, 0], todos_dados[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('Modelagem com GMM')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.colorbar(label='Clusters')
plt.grid()
plt.show()
```
<div align="center"> 
    
![figura1](../images/exercicio4cap1.png "figura 17") <legend>Figura 1.7 - Resultado ao executar código do exercício 4. Fonte: Elaborado pelo autor.
</legend>
</div>

## Exercício 5 - Janelas de Parzen
Refaça o exercício anterior com uso de Janelas de Parzen para modelagem da distribuição de probabilidade

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Configuração da semente para manter sempre os mesmos resultados
np.random.seed(123456)

# Caminho do arquivo de saída
path_out = 'saidaSim.txt'

# Parâmetros das distribuições
mu1 = [0.0, 0.5]
Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
qnt1 = 20

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
qnt3 = 50

conjMu = np.array([mu1, mu2, mu3])
conjSigma = np.array([Sigma1, Sigma2, Sigma3])
quantidades = np.array([qnt1, qnt2, qnt3])

# Criar e gravar dados
f = open(path_out, 'w')

# Listas para armazenar dados
todos_dados = []

for r in range(conjMu.shape[0]):
    dados = np.random.multivariate_normal(conjMu[r], conjSigma[r], quantidades[r])

    # Gravar dados e armazenar nos arrays
    for i in range(quantidades[r]):
        st = ','.join(map(str, dados[i]))  # Formata os dados em uma string
        st += '\n'
        f.write(st)

    # Armazenar dados
    todos_dados.append(dados)

f.close()

# Concatenar todos os dados em arrays numpy
todos_dados = np.concatenate(todos_dados)  # Todos os dados concatenados

# Aplicar o método de Janelas de Parzen
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(todos_dados)

# Geração de uma grade de pontos para visualização
x, y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
grid_points = np.vstack([x.ravel(), y.ravel()]).T

# Avaliar a densidade para cada ponto da grade
log_density = kde.score_samples(grid_points)
density = np.exp(log_density).reshape(x.shape)

# Plotar a densidade estimada
plt.figure(figsize=(10, 6))
plt.contourf(x, y, density, levels=30, cmap='viridis')
plt.scatter(todos_dados[:, 0], todos_dados[:, 1], c='red', s=10, alpha=0.5)
plt.title('Modelagem com Janelas de Parzen')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.colorbar(label='Densidade Estimada')
plt.grid()
plt.show()
```
<div align="center"> 
    
![figura1](../images/janelasdeparzen.png "figura 18") <legend>Figura 1.8 - Resultado ao executar código do exercício 5. Fonte: Elaborado pelo autor.
</legend>
</div>

## Exercício 6 - ML, Naive Bayes, mínima distância de Mahalanobis e KNN
Aplique os modelos de classificação ML, Naive Bayes, mínima distância de Mahalanobis e KNN sobre problemas de classificação envolvendo quatro classes. Realize tais aplicações sobre dados simulados. Faça as adaptações necessárias nos códigos apresentados a fim de permitir tal generalização.
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

# Configuração da semente para manter sempre os mesmos resultados
np.random.seed(123456)

# Parâmetros das distribuições
mu1 = [0.0, 0.5]
Sigma1 = [[0.75, 0.5], [0.5, 2.0]]
qnt1 = 40

mu2 = [1.0, 1.0]
Sigma2 = [[0.75, 0.5], [0.5, 2.0]]
qnt2 = 40

mu3 = [-1.5, -1.5]
Sigma3 = [[0.5, -0.5], [-0.5, 1.0]]
qnt3 = 50

mu4 = [2.0, -1.0]
Sigma4 = [[0.6, 0.2], [0.2, 0.8]]
qnt4 = 45

conjMu = np.array([mu1, mu2, mu3, mu4])
conjSigma = np.array([Sigma1, Sigma2, Sigma3, Sigma4])
quantidades = np.array([qnt1, qnt2, qnt3, qnt4])
rotulos = np.array([1, 2, 3, 4])

# Gerar dados
todos_dados = []
todos_rotulos = []
for r in range(conjMu.shape[0]):
    dados = np.random.multivariate_normal(conjMu[r], conjSigma[r], quantidades[r])
    todos_dados.append(dados)
    todos_rotulos.extend([rotulos[r]] * quantidades[r])

# Concatenar todos os dados e rótulos
X = np.concatenate(todos_dados)
y = np.array(todos_rotulos)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_acc = nb_model.score(X_test, y_test)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_acc = knn_model.score(X_test, y_test)

# Distância de Mahalanobis
mahalanobis_distances = []
for xi in X_test:
    dists = [distance.mahalanobis(xi, np.mean(X_train[y_train == c], axis=0), np.linalg.inv(np.cov(X_train[y_train == c].T))) for c in np.unique(y_train)]
    mahalanobis_distances.append(np.argmin(dists) + 1)
mahalanobis_acc = np.mean(mahalanobis_distances == y_test)

# Máxima Verossimilhança (MLE)
mle_predictions = []
for xi in X_test:
    probs = [multivariate_normal.pdf(xi, mean=np.mean(X_train[y_train == c], axis=0), cov=np.cov(X_train[y_train == c].T)) for c in np.unique(y_train)]
    mle_predictions.append(np.argmax(probs) + 1)
mle_acc = np.mean(mle_predictions == y_test)

# Exibir resultados
print(f'Naive Bayes Acurácia: {nb_acc:.2f}')
print(f'KNN Acurácia: {knn_acc:.2f}')
print(f'Distância de Mahalanobis Acurácia: {mahalanobis_acc:.2f}')
print(f'Máxima Verossimilhança Acurácia: {mle_acc:.2f}')

# Visualização dos dados
plt.figure(figsize=(10, 6))
for c in np.unique(y):
    plt.scatter(X[y == c][:, 0], X[y == c][:, 1], label=f'Classe {c}', alpha=0.6)
plt.title('Distribuição das Classes Simuladas')
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.legend()
plt.grid()
plt.show()
```
### Resultados
Acurácia por modelo:

* Naive Bayes Acurácia: 0.81
* KNN Acurácia: 0.75
* Distância de Mahalanobis Acurácia: 0.66
* Máxima Verossimilhança Acurácia: 0.72

<div align="center"> 
    
![figura1](../images/ex6cap1.png "figura 19") <legend>Figura 1.9 - Resultado ao executar código do exercício 6. Fonte: Elaborado pelo autor.
</legend>
</div>