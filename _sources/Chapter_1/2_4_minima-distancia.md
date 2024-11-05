<style>
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# 2.4: Classificação por mínima distância (casos particulares)

Levando em consideração o modelo de classificação MAP e fazendo a suposição de que as funções de verossimilhança $\textit{p}(\textbf{x}|w_{i})$ se comportam segundo a dsistribuição gaussiana multivariada, tem-se:

<div align="center">
$\begin{equation}
    \textbf{x} \sim \mathcal{N}(\mu_{i}, \Sigma_{i}), \quad \textbf{x} \in \mathbb{R}^{n}

<div align="center">
$\begin{equation}
    p(\textbf{x}|w_{i}) = \frac{1}{(2\pi)^{d/2} |\Sigma_{i}|^{1/2}} \exp\left( -\frac{1}{2} (\textbf{x} - \mu_{i})^T \Sigma_{i}^{-1} (\textbf{x} - \mu_{i}) \right) \tag{2.16}
\end{equation}$ </div>

onde:
* \(\textbf{x}\) é o vetor de características,
* \(\mathcal{N}(\mu_{i}, \Sigma_{i})\) denota a distribuição normal multivariada com média \(\mu_{i}\) e matriz de covariância \(\Sigma_{i}\)
* \(\textbf{x} \in \mathbb{R}^{n}\) indica que \(\textbf{x}\) pertence ao espaço de dimensão \({n}\),
* \(\mu_{i}\) é o vetor de média para a classe \(w_{i}\),
* \(\Sigma_{i}\) é a matriz de covariância para a classe \(w_{i}\),
* \(d\) é a dimensão do vetor \(\textbf{x}\).

Diante do modelo matemático acima e recapitulando que a reinterpretação da regra de decisão, incialmente expressa por comparações entre distribuições de probabilidades, na forma de funções discriminantes, implica na composição de $p(\textbf{x}|w_{i})$ em uma função monotônica crescente ou decrescente $f:\mathbb{R} \to \mathbb{R}$, por convenivência ao adotar $f(x) = \ln(x)$, obtem-se:

<div align="center">
$\begin{equation}
    g_{i}(\textbf{x}) = \ln(p(\textbf{x}|w_{i}))P(w_{i}) = \ln(p(\textbf{x}|w_{i})) + P(w_{i})  \tag{2.17}
\end{equation}$ </div>

A função discriminante $g_{i}(\textbf{x})$ combina a evidência da classe $w_{i}$ dada a observação $\textbf{x}$ e a probabilidade $\textit{a priori}$ dessa classe. Essa função é utilizada para comparar diferentes classes que maximiza a probabilidade $\textit{a posteriori}$. Desenvolvendo a Equação 2.17 e expressando as parcelas que independem de $\textbf{x}$, pode-se concluir que a expressão obtida para $g_{i}(\textbf{x})$ é uma função discriminante quadrática. Essa expressão compreende o caso geral de função discriminante que decorre da regra de classificação de Bayes com distribuição guassiana multivariada.

Considerando $x \sim N(\mu_i, \Sigma_i)$ independentemente das classes inseridas no problema de classificação e admitindo que as variâncias e covariâncias possuem comportamento idêntico e, ainda, admitindo que $\Sigma$ não é diagonal, implicando a existência de coravirâncias entre as componentes $\textbf{x}$, têm-se que:

<div align="center">
$\begin{equation}
    g_i(\textbf{x}) = -\frac{1}{2} \textbf{x}^T \Sigma^{-1} \textbf{x} + \frac{1}{2} \textbf{x}^T \Sigma^{-1} \boldsymbol{\mu}_i + \frac{1}{2} \boldsymbol{\mu}_i^T \Sigma^{-1} \textbf{x} 
    - \frac{1}{2} \boldsymbol{\mu}_i^T \Sigma^{-1} \boldsymbol{\mu}_i - \frac{1}{2} \ln(2\pi)^n - \frac{1}{2} \ln|\Sigma| + \ln P(w_i)  \tag{2.18}
\end{equation}$ </div>

No entando, pode-se verificar que independente de $\textbf{x}$, as parcelas $- \frac{1}{2} \ln(2\pi)^n - \frac{1}{2} \ln|\Sigma| + \ln P(w_i)$ são constantes. Segundo a suposição inicial, as classes são equiprováveis, tornando $\ln P(w_i)$ constante de $\textbf{x}^T \Sigma^{-1}\textbf{x}$ invariante à mudança de classes, logo sendo irrelevante para o processo de classificação. Sendo assim:

<div align="center">
$\begin{equation}
    g_i(\textbf{x}) = \textbf{x}^T \Sigma^{-1} \boldsymbol{\mu}_i - \frac{1}{2} \ln|\Sigma| \tag{2.19}
\end{equation}$ </div>

Após a manipulação algébrica da Equação 2.19, obtêm-se a forma equivalente $g_i(\textbf{x}) = -\frac{1}{2}\left((\textbf{x} - \boldsymbol{\mu}_i)^T \Sigma^{-1} (\textbf{x} - \boldsymbol{\mu}_i)\right)^{\frac{1}{2}}$, similar à medida conhecida como "distância de Mahalonobis"[^1]. Por este motivo, as considerações feitas sobre as variÂcnias e covariâncias denominam a função de discriminação obtida como "classificador de mínima distância de Mahalanobis", cuja regra de decisão associada é:

<div align="center">
$\begin{equation}
    \((x, w_i) \leftrightarrow \arg \max_{j=1,\ldots,c} g_j(x) \equiv \arg \min_{j=1,\ldots,c} d_{m_j}(x)\) \tag{2.20}
\end{equation}$ </div>

sendo $dm_i(x) = -g_i(\textbf{x})$.

Agora, considerando que $\Sigma$ é diagonal, levando à suposição de covariâncias nulas ou de independência entre as componentes de $\textbf{x}$, e que variâncias são idênticas, ou seja, $\Sigma = \sigma^2\textbf{I}$. Consequentemente:

<div align="center">
$\begin{equation}
    g_i(\textbf{x}) = -\frac{1}{2}\left((\textbf{x} - \boldsymbol{\mu}_i)^T \sigma^2\textbf{I} (\textbf{x} - \boldsymbol{\mu}_i)\right)^{\frac{1}{2}}
    = -\frac{1}{\sigma^2\textbf{I}}\left((\textbf{x} - \boldsymbol{\mu}_i)^T (\textbf{x} - \boldsymbol{\mu}_i)\right)^{\frac{1}{2}} =
    = -\frac{1}{\sigma^2\textbf{I}}\|\| x - \mu_i \|\|
    \tag{2.21}
\end{equation}$ </div>

Uma vez que a constante $\frac{1}{\sigma^2\textbf{I}}$ não altera em função da mudança de classes, é possível removê-ça da expressão. Dessa forma, a equação 2.21 revela que o processo de classificação é guiado pela distância euclidiana $de_{i}(\textbf{x}) = -g_i(\textbf{x})$, de modo similar à Equação 2.20.