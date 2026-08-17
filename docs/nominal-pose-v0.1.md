# Pose nominal do Digital Twin v0.1

Os valores desta nota são **HIPÓTESES TEMPORÁRIAS** da Etapa 3.5. Eles
definem o significado da ação neutra e não afirmam que o robô consegue se
equilibrar passivamente.

## Ângulos nominais

| Articulação | Esquerda | Direita | Justificativa |
| --- | ---: | ---: | --- |
| Hip roll | 0° | 0° | Mantém a pelve nivelada no plano frontal. |
| Hip pitch | -5° | -5° | Inicia uma flexão sagital pequena. |
| Knee pitch | 10° | 10° | Evita uma configuração de joelho totalmente estendido. |
| Ankle roll | 0° | 0° | Mantém as solas niveladas lateralmente. |
| Ankle pitch | -5° | -5° | Compensa hip e knee pitch para deixar o pé horizontal. |

Em cada perna, `hip_pitch + knee_pitch + ankle_pitch = 0°`. Com o tronco
preso rigidamente à pelve e o `freejoint` sem rotação inicial, isso mantém o
tronco vertical e os pés paralelos ao chão.

A altura inicial da raiz, `0,292125 m`, também é uma **HIPÓTESE TEMPORÁRIA**.
Ela foi calculada a partir das dimensões atuais da coxa, canela e pé para
posicionar a face inferior das solas aproximadamente em `z=0`, sem alterar
geometrias, massas, gravidade, atrito ou centro de massa.

## Mapeamento da ação

A ação continua normalizada em `[-1, 1]`. O deslocamento é calculado ao redor
da pose nominal usando escalas diferentes para cada direção:

```text
action < 0: target = nominal + action * (nominal - lower_limit)
action > 0: target = nominal + action * (upper_limit - nominal)
action = 0: target = nominal
```

As amplitudes não introduzem novos limites hipotéticos: elas são derivadas
dos ranges físicos atuais de cada junta. O target final ainda é limitado à
interseção entre o range da junta e o `ctrlrange` do atuador.
