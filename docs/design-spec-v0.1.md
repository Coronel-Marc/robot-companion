# Robot Companion Design Specification v0.1

## 1. Visão do Projeto

O Robot Companion é um robô humanoide de aproximadamente 50 cm, projetado para pesquisa em aprendizado por reforço, robótica e interação humano-robô.

O objetivo principal não é apenas construir um robô capaz de andar, mas desenvolver uma plataforma modular para pesquisa e experimentação que evolua continuamente.

## 2. Filosofia do Projeto

- O MuJoCo será tratado como um gêmeo digital do robô físico.
- Toda peça estrutural deverá existir primeiro na simulação.
- O hardware deverá acompanhar o modelo virtual.
- O projeto será totalmente modular.
- Facilidade de manutenção é prioridade.

## 3. Objetivos

O robô deverá ser capaz de...

- permanecer em pé;
- caminhar;
- levantar após uma queda;
- manipular objetos leves;
- interagir verbalmente;
- aprender novos comportamentos.

## 4. Requisitos Mecânicos

- Altura:

  - ~50 cm

- Construção:

  - impressão 3D

- Arquitetura:

  - endoesqueleto estrutural + carenagem removível

- Centro de massa:

  - preferencialmente baixo

## 5. Aparência

  Incluir:

    - imagens.

    - Referências.

    - Paleta de cores.

    - Formato do corpo.

    - Inspirações.

## 6. Arquitetura de Software

MuJoCo

↓

Gymnasium

↓

Stable Baselines3

↓

Modelo treinado

↓

Robô físico

## 7. Roadmap

Fase 1

✓ Ambiente Python

✓ MuJoCo

✓ Primeira perna

✓ Primeiro bípede — Digital Twin v0.1, Etapas 1–6

⬜ RL para equilíbrio

⬜ Caminhada

⬜ Braços

⬜ Cabeça

⬜ Manipulação

## 8. Decisões em aberto

Ainda não foi escolhido o servo.

Ainda não foi escolhido o computador embarcado.

Ainda não sei se o pescoço terá 2 ou 3 DOF.

## 9. Diário de decisões

````

2026-08-06

Decidido utilizar arquitetura em camadas.

Motivo:

Facilidade de manutenção.

Separação entre estrutura e estética.

Compatibilidade com Digital Twin.

````
