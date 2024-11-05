<style>
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# 1.2: Naive Bayes

Como discutido anteriormente, um padrão $\textbf{x} \in \textbf{X}$ é interpretado como um evento aleatório. Entretanto uma vez que $\textbf{X}$ se refere a um espaço $\textit{n}$-dimensional qualquer, é possível reinterpretar $\textbf{x}$ como $\textit{n}$ para eventos aleatórios $x_{1},..., x_{n}$, os quais podem, ou não, ser independentes entre si.
Nas diversas aplicações envolvendo classificação ou regressão de dados, o estabelecimento de modelos que não levem em consideração as estruturas de correlação entre tais variáveis pode proporcionar resultados inadequados. No entanto, um modelo "ingênuo" admite $x_{1},..., x_{n}$ independentes entre si, mesmo que não o sejam, e admite que contribuem igualmente para o resultado. Sob essa condição, têm-se 

<div align="center">

$\begin{equation}
    p(\textbf{x}|ω_{i}) = p(x_{1},..., x_{n}|ω_{i}) = p(x_{1}|ω_{i})p(x_{2}|ω_{i})...p(x_{n}|ω_{i}) \tag{2.12}
\end{equation}$ </div>

Sendo assim, a regra de decisão estabelecida na Equação 2.2 torna-se equivalente a:

<div align="center">

$\begin{equation}
    (\textbf{x},ω_{j}) \leftrightarrow arg_{ω_{j}\inΩ} max \prod_{i=1}^{n} p(x_{i}|ω_{j}) \tag{2.13}
\end{equation}$ </div>

O modelo Naive Bayes simplifica problemas de classificação ao assumir que todas as variáveis são condicionalmente independentes segundo a classe. A simplificação reduz significativamente a complexidade do modelo, pois requer o cálculo de apenas uma probabilidade para cada variável, em vez de considerar todas as combinações possíveis. Como resultado, o Naive Bayes é computacionalmente eficiente e oferece bom desempenho em tamanhos de amostra pequenos.