<style>
    legend {
        font-size: 16px;
    }
    main {
        text-align: justify;
    }
</style>

# Capítulo 3: Classificadores não lineares

Na seção anterior, foram apresentados métodos capazes de distinguir entre pares de classes linearmente separáveis. A aplicação desses métodos a problemas que envolvem classes não linearmente separáveis não foi limitada, porém, sabe-se que erros podem ocorrer durante o processo de classificação. No entanto, existem situações em que tentar separar duas classes por meio de uma superfície de decisão linear é inviável. 

Classificadores não lineares englobam algoritmos de aprendizado de máquina que são capazes de modelar relações entre as classes de entrada e saída quando estas não podem ser separadas de maneira simples por uma linha reta ou um plano. As fronteiras de decisão que determinam a separação das classes adaptam-se à estrutura dos dados de forma mais flexível do que aquelas nos classificadores lineares discutidos no capítulo anterior. Este capítulo discutirá alguns algoritmos de classificadores não lineares de forma a introduzir o leitor aos conceitos e teorias acerca de classificadores multicamadas, a forma de treiná-los e o ajuste de quaisquer parâmetros que sejam necessários.

A Figura 3.1 ilustra problemas desse tipo. Uma maneira eficaz de lidar com essas dificuldades é utilizando classificadores não lineares.

<div align="center"> 

![figura31](../images/figura31.png "figura 3.1") <legend>Figura 3.1 - Casos que inviabilizam a separação entre classes de forma linear.</legend> </div>