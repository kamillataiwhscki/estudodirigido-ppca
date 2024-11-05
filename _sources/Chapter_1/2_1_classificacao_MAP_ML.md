<style>
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# 2.1: Classificação MAP e ML

Segundo a Equação 2.2, que estabelece um padrão segundo a maximização da probabilidade $\textit{a priori}$ e a Equação 2.1 que relaciona as probabilidades $\textit{a priori}$, $\textit{a posteriori}$ e a função de verossimilhança, $P(ω_{i}|\textbf{x}) > P(ω_{j}|\textbf{x})$, para  ${i}\neq{j} = {1},..., {c}$, implica na associação de ${x}$ à classe  $ω_{i}$. Por equivalência $\frac{P(\textbf{x}|ω_{i})P(ω_{i})}{\text{p(\textbf{x})}} > \frac{P(\textbf{x}|ω_{j})P(ω_{j})}{\text{p(\textbf{x})}} $ implica na mesma associação.
Com base nesta relação de comparação, nota-se que a $\textit{evidência} p(\textbf{x})$ não interfere na escolha da classe que maximiza a probabilidade $\textit{a posteriori}$. Sendo assim, possível obter o modelo de classificação  $\textit{maximum a posteriori} (MAP)$:

<div align="center">
$\begin{equation}
    (\textbf{x},ω_{j}) \leftrightarrow arg_{ω_{j}\inΩ} max p(\textbf{x}|ω_{j})P(ω_{j}) \tag{2.10}
\end{equation}$ </div>

Por sua vez, caso se assuma equiprobabilidade das classes, o modelo MAP fica restrito às funções de verossimilhanças que modelam as diferentes classes. Desta simplificação surge o modelo ed classificação  $\textit{maximum likelihood} (ML)$, sendo o modelo de máxima verossimilhança:

<div align="center">
$\begin{equation}
    (\textbf{x},ω_{j}) \leftrightarrow arg_{ω_{j}\inΩ} max p(\textbf{x}|ω_{j}) \tag{2.11}
\end{equation}$ </div>

O modelo ML e é útil quando não se tem informações confiáveis sobre as probabilidades $\textit{a priori}$ das classes ou quando todas as classes são igualmente prováveis.