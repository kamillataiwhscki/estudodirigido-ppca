<style>
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# 2.5: Estimação de funções densidade de probabilidade

A função de densidade de probabilidade (denominada anteriormente de função de verossimilhança) deve ser empregada de acordo com o comportamento (ou suposição deste) dos dados a serem classificados ou mesmo a partir de uma exigência imposta para formalização do método de classificação. 

Na possibilidade do emprego de um modelo de distribuição específica, o ajuste deste modelo ao problema específico se dá através da escolha dos parâmetros. Dentre as diferentes técnicas existentes para a obtenção de tais parâmetros, a estimação por $\textit{máxima verossimilhança}$ é uma alternativa, sendo possível, através dela, obter expressões para os estimadores dos parâmetros de uma dada distribuição de probabilidade a partir do seu modelo.

No entando em casos de distribuições conhecidas gerarem violações e/ou considerações irreais a respeito de dados, pode-se utilizar "misturas" entre distribuições de probabilidade conhecidas ou mesmoe stimar a distribuição de probabilidade sem suposição sobre um dado modelo e a respectiva parametrização.

# 2.5.1: O conceito da estimação por máxima verossimilhança

A função de verossimilhança é representada por $\textit{p}(\textbf{x}|w_{i})$, sendo a função dos padrões $\textbf{x}$ em relação à classe $w_{i}$, isso corresponte a uma distribuição de probabilidade que modela o comportamento de $\textbf{x}|w_{i}$. Por sua vez, tal distribuição pode ser expressa como $\textit{p}(\textbf{x}|\theta_{1}, \ldots, \theta_k)$, em que $\theta_{1},\ldots, \theta_k$ são parâmetros que tornam a distribuição de probabilidade em questão aderente ao comportamento de $w_{i}$

Nas condições apresentadoas, a problemática reside em determinar os valores $\theta_{1},\ldots, \theta_k$ que tornam $\textit{p}(\textbf{x}|\theta_{1}, \ldots, \theta_k)$ ajustada à realidade de $w_{i}$. Segundo Fisher (1992) os parâmetros ótimos que modelam $\textit{p}(\textbf{x}|\theta_{1}, \ldots, \theta_k)$, de acordo com um conjunto de observações $D = \{x_1, \ldots, x_m\}$, correspondem aos mesmos parâmetros que maximizam a seguinte função:

<div align="center">
$\begin{equation}
    L(\theta_i | D) = p(\theta_i | x_1) p(\theta_i | x_2) \ldots p(\theta_i | x_m) = \prod_{j=1}^{m} p(x_j | \theta_i) \tag{2.21}
\end{equation}$ </div>

equivalentemente:

<div align="center">
$\begin{equation}
    \hat{\theta}_i = \underset{\theta_i \in \Theta}{\arg \max} \prod_{j=1}^{m} p(x_j | \theta_i) \tag{2.22}
\end{equation}$ </div>

A fim de viabilizar o desenvolvimento matemático, a função de verossimilhança é substituída pelo seu logarítmo natural, uma vez que ambas as formas compartilham o mesmo ponto máximo, visto que o $\ln(\cdot)$ é uma função monotônica estritamente crescente. Utilizando a regra da cadeia, importante ferramenta do Cálculo Diferencial, é possível resolver o problema de maximização apresentado anteriomente pelo valor de $\theta_i$, que anula a derivada $L(\theta_i | D)$ com relação a este mesmo parâmetro, ou seja:

<div align="center">
$\begin{equation}
   \hat{\theta}_i = \frac{\partial \ln(L(\theta_i | D))}{\partial \theta_i} \Bigg|_{\theta_1, \ldots, \theta_k} = 0 \tag{2.23}
\end{equation}$ </div>

Sabendo que $\ln(L(\theta_i | D)) = \ln(L(\theta_i | D)) = \sum_{j=1}^{m} \ln(p(x_j | \theta_i))$, têm-se:

<div align="center">
$\begin{equation}
    \frac{\partial \ln(L(\theta_i | D))}{\partial \theta_i} = \sum_{j=1}^{m} \frac{\partial \ln(p(x_j | \theta_i))}{\partial \theta_i} = \sum_{j=1}^{m} \frac{1}{p(x_j | \theta_i)} \frac{\partial p(x_j | \theta_i)}{\partial \theta_i} \tag{2.24}
\end{equation}$ </div>

Desse modo, cada $\hat{\theta}_i$, para $i = 1, \ldots, k$, a ser escolhido deve anilar o lado direito da Equação 2.24. Segundo Mood et al. (1974, caput Negri (2021)), ao passo que o número de observações em $D$ aumenta, o estimado de máxima verossimilhança gera uma aproximação assintótica, ou seja, cada vez mais próxima, ao parâmetro verdadeiro.

# 2.5.2 Modelo ed mistura de gaussianas e o algoritmo EM

Combinações de diferentes distribuições de probabilidade conhecidas dão origem a novas distribuições de probabilidade. Nesse contexto, o chamado modelo de mistura expressa a modelagem de uma distribuição $p(\textbf{x})$ como resultado da combinação linear entre funções de densidade de probabilidade. Dessa forma:

<div align="center">
$\begin{equation}
   p(x) = \sum_{j=1}^{k} \lambda_j p(x | \theta_j) \quad 
   \text{com} \quad \sum_{j=1}^{k} \lambda_j = 1; \quad \int_{x \in X} p(x | \theta_j) \, dx = 1 \tag{2.25}
\end{equation}$ </div>

em que $\theta_j$ corresponde a uma tupla de parâmetros exigidos pela $j$-ésima componente. A combinação apresentada na Equação 2.25 envolve $k$ distribuições $\textit{p}(\textbf{x}|\theta_j)$ podenradas pelos respectivos $\lambda_j \in [0,1]$, com $j = 1,\ldots, k$, sendo possível observar a condição $\int_{x \in X} p(x | \theta_j) \, dx$ é atendida, visto que as diferentes componentes atendem a tal condição e a ponderação efetuada por  $\lambda_j$ não provoca contração ou expansão do intervalo $[0,1]$. 

Um modelo de mistura é capaz de aproximar qualquer distribuição desde que seja considerada uma quantidade $k$ adequada de componente devidamente parametrizadas e ponderadas. Nesse sentido, o espaço de parâmetros que contém a solução do problema proposto possui como elementos tuplas da forma $\Psi = (\lambda_{1}, \ldots, \lambda_k, \theta_{1}, \ldots, \theta_k)$.

De modo a colocar o método da máxima verossimilhança em ação e obter a estimação dos parâmetro, o problema seria reduzido à maximização da função $L(\Psi) = \prod_{i=1}^{m} \sum_{j=1}^{k} \lambda_j p(x_i; \theta_j)$. No entando, a não linearidade deste problema não permite que o ponto seja obtido atravás da derivada de $L(\Psi)$ utilizando a regra da cadeia. Como alternativa, é feito o uso de um processo iterativo chamado $\textit{Expectation-Maximization}$ (EM).

Considerando a Equação 2.25, têm-se:

<div align="center">
$\begin{equation}
    p(x | \Psi) = \sum_{j=1}^{k} \lambda_j p_j(x | y, \theta_j) \tag{2.26}
\end{equation}$ </div>

em que $p_j(x | y, \theta_j)$ é uma das $k$ componentes da mistura com parâmetro $\theta_j$; e $y=[y_{1},\ldots,y_{k}]$ compreende um vetor binário, tal que apenas uma das componentes equivale a 1 e as demais são nulas, cuja finalidade é identificar as componentes da mistura responsáveis por gerar $\textbf{x}$. Sobre os pesos que podenram a mistura é razoável admitir $\lambda_j = p(y_{j})$, uma vez que este expressa a probabilidade de $\textbf{x}$ ter sido gerado pela $j$-ésima componente.

Admitindo $_j(x | y, \theta_j)$ como gaussianas multivariadas torna-se possível denominar o processo a seguir como Modelo de Mistura de Gaussianas ($\textit{Gaussian Mixture Model}$ - GMM). Com base nas observações de $D = \{x_1, \ldots, x_m\}$ e considerando uma configuração de parâmetros $\Psi = (\lambda_{1}, \ldots, \lambda_k, \theta_{1}, \ldots, \theta_k)$ é possível calcular a pertinência de cada $\textbf{x}_{i}$, com $i = 1,\ldots, m$, em relação à componente $j$, através da expressão:

<div align="center">
$\begin{equation}
    w_{ij} = p(y_j = 1 | x_i, \psi) = \frac{\lambda_j p_j(x_i | y, \theta_j)}{\sum_{l=1}^{k} \lambda_l p_l(x_i | y, \theta_l)} \tag{2.27}
\end{equation}$ </div>

Uma vez postas essas considerações, têm-se condições necessárias para inciar um processo iterativo que converge à configuração de parâmetros que conduz a uma distribuição de probabilidade modelada a partir da mistura de gaussianas. Tal processo compreende duas etaapas, uma sobre a Esperança (E - $\textit{Expectation}$) com que cada componente atua na mistura e outra sobre a Maximização (M - $\textit{Maximization}$)da probabilidade através da atualização dos parâmetros.

# 2.5.3 Método do histograma

O método do histograma consiste em uma abordagem livre de parâmetros de distribuições de probabilidade, ou seja, não paramétrica, sendo forma simples de obter uma aproximação para a função de distribuição de probabilidade referente ao comportametno de um dado conjunto de observações $D = \{ x_i \in \mathbb{R} \, | \, i = 1, \ldots, m \}$. Para tal, o intervalo de valores que contém as observações é particionado em subintervalos disjuntos de amplitude $h$, cujo significado é similar aos intervalos ed calsses em uma tabela de distribuição de frequência usada na construção de histograma. Por conseguinte: 

<div align="center">
$\begin{equation}
    \hat{p}(x) = \frac{1}{h} \frac{q(x)}{m} \tag{2.28}
\end{equation}$ </div>

sendo $q(x)$ uma quantificação a respeito do número de observações de $D$ e que ocupam o mesmo subintervalo que contém $x$. Sendo importante destacar que este método proporciona $\hat{p}(x)$ convergente à verdadeira distribuição de probabilidade $p(x)$, desde que: $m$ seja suficientemente grande; $h$ seja suficientemente pequeno; $q(x) \to \infty; \quad \frac{q(x)}{m} \to 0$. Além disso, esta proposta contempla apenas dados unidimensionais.

# 2.5.4 Janelas de Parzen

De forma a contemplar a aproximação não paramétrica sobre a distribuição de probabilidade $D = \{ x_i \in \mathbb{R}^n : i = 1, \ldots, m \}$, utiliza-se hipercubos de lado $h$. visto que utilizar intervalos de amplitudes $h$ seria impossivel. Além dessa consideração, é empregada a seguinte função:


<div align="center">
$\begin{equation}
    \phi(x_i) = 
\begin{cases} 
1 & \text{se } |x_{ij}| < \frac{1}{2}, \; j=1,\ldots,n \\ 
0 & \text{caso contrário} 
\end{cases} \tag{2.28}
\end{equation}$ </div>

$\phi(.)$ atua como um verificador lógico cujo retorno igual a 1 indica que $x_{i}$ é interno ao hipercubo de lado unitário centrado na origem do espaco $\mathbb{R}^n$. Dessa forma, a substituição do argumento $x_{i}$ por $\left(\frac{x_i - x}{h}\right)$ torna-se equivalente a uma tranlação que move o hipercubo da origem ao ponto $\textbf{x}$ seguida por transformação em escala, tornando-o análogo a um hipercubo de lado $h$.

Por sua vez, a aproximação desejada é expressa por:

<div align="center">
$\begin{equation}
    \hat{p}(x) = \frac{1}{h^n m} \sum_{i=1}^{m} \phi \left(\frac{x_i - x}{h}\right) \tag{2.29}
\end{equation}$ </div>

Um problema relacinado à Equação 2.29 refere-se a sua descontinuidade, ou seja, o comportamento binário desempenhado por $\phi(.)$ pode proporcionar mudanças repentinas entre valores de probabilidade aproximados e valores nulos, causados pela ausência de observações emd eterminadas regiões não abrangidas por $D$. Uma alternativa consiste em adotar outra formulação para $\phi(.)$ que seja capaz de realizar ponderações suaves sobre o espaço dos dados e assegure $\int_{\mathbb{R}^n} \phi(x) \, dx = 1$. Para tal propósito, é usalmente adotada a função:

<div align="center">
$\begin{equation}
    \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^T x}{2}} \tag{2.30}
\end{equation}$ </div>

# 2.5.5 $K$-vizinhos mais próximos

Na formulação inicial sobre o método Janelas de Parzen, a estimação da distribuição de probabilidade para um dado $\textbf{x}$ é obtido através da noção do número de observações em $D$ que ocorrem no interior de um hipercubo de volume fixo. Uma forma reversa de alcançar o mesmo resultado é fixando um número $k \in \mathbb{N}$  de observações, com $k < |D|$, sobre os quais é determinado o volume da menor hiperesfera que contém $k$ observações. Desse modo, ao passo que regiões com baixa densidade de probabilidade serão associadas a grandes volumes, as regiões de alta densidade serão compreendidas por volumes restritos. Sendo expressa por:

<div align="center">
$\begin{equation}
    \hat{p}(x) = \frac{k}{mV_{k}(x)} \tag{2.31}
\end{equation}$ </div>

em que $m$ é a cardinalidade de $D$ e $V_{k}(x)$ corresponde ao volume da menor hiperesfera possível com centro em $\textbf{x}$ e que contém $k$ observações de $D$.

A quantidade $V_{k}(x)$ é facilmente obtida através de:

<div align="center">
$\begin{equation}
   V_k(x) = V_0 \rho^n \quad \text{com} \quad V_0 =  
\begin{cases} 
\frac{\pi^{n/2}}{(n/2)!} & \text{se } n \text{ par} \\ 
\frac{2^n \pi^{(n-1)/2}}{((n-1)/2)! n!} & \text{se } n \text{ ímpar} 
\end{cases} \tag{2.32}
\end{equation}$ </div>

sendo $\rho$ a $k$-ésima menor distância observada entre $\textbf{x}$ e os elementos de $D$.

Considerando o conjunto de observações $D = \{ (x_i, y_i) : i = 1, \ldots, m \}$, no qual $x_i$ está associado a uma classe de \Omega = \{ w_1, \ldots, w_c \} através de indidcadores de classe $y_i \in Y = 1,\ldots,c$. A partir desse conjunto é possível estimar densidades de probabilidade para cada uma das classes isoladamente através da seguinte expressão:

<div align="center">
$\begin{equation}
    p(x | w_j, k) = \frac{k}{m_j V_0 \rho_{kj}(x)} \tag{2.33}
\end{equation}$ </div>

sendo $\rho_{kj}(x)$ a $k$-ésima menor distância observada entre $\textbf{x}_{i}$ e cada uma das $m_j$ observações presentes em $D$ e associadas à classe $w_{j}$.

Nestas condições, com base no modelo ML, é desenvolvida a seguinte relação:

<div align="center">
$\begin{equation}
   (x, w_j) \quad \leftrightarrow \quad p(x | w_j, k) > p(x | w_l, k) \quad \\
   \leftrightarrow \quad \frac{k}{m_j V_0 \rho_{kj}(x)} > \frac{k}{m_l V_0 \rho_{kl}(x)} \quad \\
   \leftrightarrow \quad m_l \rho_{kl}(x) > m_j \rho_{kj}(x) \tag{2.34}
\end{equation}$ </div>

Interpretando $d_k^{(j)}(x) = m_j \rho_{kj}(x)$ como uma medida de distância, têm-se:

<div align="center">
$\begin{equation}
   (x, w_j) \leftrightarrow \arg_{j=1, \ldots, c} \min d_{k_j}(x) \tag{2.35}
\end{equation}$ </div>

como expressão da regra de decisão que caracteriza o classificador $k$-vizinhos mais próximos $\textit{K-Nearest Neighbors - KNN}$.
