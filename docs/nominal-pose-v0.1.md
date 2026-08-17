# Pose nominal do Digital Twin v0.1

Os valores desta nota são **HIPÓTESES TEMPORÁRIAS** da Etapa 3.5. Eles
definem o significado da ação neutra e não afirmam que o robô consegue se
equilibrar passivamente.

O registro canônico de classificação e rastreabilidade desses valores está em
`docs/digital-twin-assumptions-v0.1.md`.

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

A altura inicial da raiz, `0,292125 m`, é um **VALOR DERIVADO** das hipóteses
de geometria e pose atuais. Ela foi calculada para posicionar a face inferior
das solas aproximadamente em `z=0`, sem alterar geometrias, massas, gravidade,
atrito ou centro de massa. Se essas hipóteses mudarem, a altura deve ser
recalculada.

Esse valor não é o mesmo que o `pos="0 0 0.303"` do `pelvis_root` no MJCF. O
estado cru do XML usa todas as juntas em `0°` e deixa as solas `0,010 m` acima
do piso. O ambiente aplica a pose nominal flexionada e sobrescreve a altura para
apoiar as solas. A diferença é esperada entre esses dois contextos e está
derivada em `docs/digital-twin-assumptions-v0.1.md`; nenhum dos valores deve ser
alterado isoladamente sem revisar pose, geometria e testes.

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
