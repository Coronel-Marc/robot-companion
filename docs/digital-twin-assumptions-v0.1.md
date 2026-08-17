# Hipóteses e rastreabilidade do Digital Twin v0.1

Estado consolidado em 2026-08-17, após as Etapas 1–5. Este documento registra o
estado executável atual; ele não transforma valores simulados em especificações do
hardware. Alterar este arquivo, por si só, não altera a simulação.

## 1. Como ler este registro

Cada item rastreável tem um identificador e uma das classificações abaixo:

- **Decisão de projeto (DP):** direção intencional adotada pelo projeto. Pode ser
  revisada, mas não é apenas um número experimental sem contexto.
- **Hipótese temporária (HT):** escolha provisória necessária ao desenvolvimento;
  exige calibração, experimento ou confirmação no hardware.
- **Valor derivado (VD):** resultado calculado a partir do código ou do MJCF atual;
  muda se sua fonte executável mudar.
- **Decisão em aberto (DA):** definição deliberadamente adiada.

Contagem deste documento: **66 hipóteses temporárias**, **12 decisões de projeto**,
**16 valores derivados** e **28 decisões em aberto**. A contagem considera cada ID
único como um item, mesmo quando um item cobre componentes simétricos esquerdo e
direito.

## 2. Fontes da verdade

| Fonte | Responsabilidade | Classificação |
| --- | --- | --- |
| `docs/design-spec-v0.1.md` | Requisitos, visão e intenção do robô físico. | DP-012 — decisão de projeto |
| Este documento | Hipóteses experimentais e fotografia rastreável da simulação. | DP-012 — decisão de projeto |
| Código Python e MJCF | Comportamento executável e valores efetivamente usados. | DP-012 — decisão de projeto |

Em caso de divergência numérica, código/MJCF descrevem o comportamento atual e a
divergência deve ser corrigida ou registrada aqui. Quando uma hipótese se tornar
definitiva, a especificação de alto nível deve ser atualizada quando aplicável.

## 3. Decisões de projeto vigentes

| ID | Decisão e origem | Justificativa | Onde vive |
| --- | --- | --- | --- |
| DP-001 | Estatura final alvo de aproximadamente `0,50 m`. | Escala escolhida para a plataforma física de pesquisa; não é a altura do modelo parcial atual. | `docs/design-spec-v0.1.md` |
| DP-002 | MuJoCo é o gêmeo digital e a estrutura física deve acompanhar o modelo virtual. | Manter rastreabilidade entre simulação e hardware. | `docs/design-spec-v0.1.md` |
| DP-003 | Um MJCF canônico único para o Digital Twin v0.1. | Reduzir complexidade durante a primeira versão. | `assets/mujoco/models/digital_twin_v0_1.xml` |
| DP-004 | Escopo físico atual: pelve, tronco e duas pernas; sem braços, pescoço ou cabeça. | Incluir somente o necessário para postura e equilíbrio inicial. | MJCF canônico |
| DP-005 | Raiz livre e 12 hinges articulares, com quadril na ordem yaw/roll/pitch e tornozelo roll/pitch. | Manter cada DOF explícito e inspecionável. | MJCF canônico |
| DP-006 | Dez juntas recebem atuadores de posição; Hip Yaw L/R existem, mas não recebem atuador nesta fase. | Implementação incremental do primeiro bípede. | MJCF canônico e `BipedalTwinEnv.ACTUATED_JOINT_NAMES` |
| DP-007 | Ação normalizada em `[-1, 1]`, centrada na pose nominal. | Separar interface do agente dos ranges físicos e dar significado estável a `action=0`. | `src/companion_robot/envs/bipedal_twin_env.py` |
| DP-008 | Reset usa `self.np_random` e a semântica de seed do Gymnasium. | Reprodutibilidade sem RNG global. | `BipedalTwinEnv.reset()` |
| DP-009 | Gravidade terrestre uniforme `(0, 0, -9,81) m/s²`. | Representar operação terrestre nominal. | MJCF canônico |
| DP-010 | Integrador `implicitfast`. | Escolha atual de integração estável e eficiente; ainda requer validação de fidelidade. | MJCF canônico |
| DP-011 | Ângulos do MJCF expressos em graus e limites automáticos habilitados (`autolimits=true`). | Legibilidade do XML e ativação explícita de ranges. | MJCF canônico |
| DP-012 | Separação de fontes da verdade descrita na seção 2. | Evitar que documentação experimental substitua requisitos ou implementação. | Documentação do projeto |

## 4. Geometria e proporções atuais

### 4.1 Dimensões simuladas

Eixos: `x` longitudinal, `y` lateral e `z` vertical. Em boxes, os valores abaixo
são dimensões completas; o MJCF armazena meias dimensões. Em cápsulas, “entre
centros” vem de `fromto`, e o comprimento de colisão inclui duas extremidades de
raio. Todas estas dimensões são hipóteses da simulação, não desenhos finais de
fabricação.

| ID | Elemento | Dimensão/posição atual | Origem e justificativa | Validar no hardware | Classificação |
| --- | --- | --- | --- | --- | --- |
| HT-001 | Altura crua da raiz no MJCF | `z=0,303 m` | Com todas as juntas em `0°`, deixa as solas `0,010 m` acima do piso; o ambiente substitui esse estado cru no reset. | Decidir futuramente se o MJCF cru deve continuar representando uma pose suspensa. | Hipótese temporária |
| HT-002 | Núcleo da pelve (`pelvis_core_geom`) | box `0,050 × 0,060 × 0,030 m` | Volume central simples para a raiz livre. | Espaço de fixação e estrutura. | Hipótese temporária |
| HT-003 | Pelve (`pelvis_geom`) | box `0,110 × 0,180 × 0,050 m` | Pelve deliberadamente larga para futuros componentes. | Dimensões internas, ergonomia e interferências. | Hipótese temporária |
| HT-004 | Tronco | centro `0,085 m` acima da pelve; box `0,090 × 0,130 × 0,120 m` | Bloco compacto para representar carga superior. | Volume de bateria/computador e carenagem. | Hipótese temporária |
| HT-005 | Eixos dos quadris | `y=±0,062 m`, `z=-0,025 m`; distância lateral `0,124 m` | Separação compatível com a pelve larga. | Largura de atuadores e folgas. | Hipótese temporária |
| HT-006 | Módulo Hip Yaw | cilindro, raio `0,022 m`, altura `0,024 m` | Volume simples de articulação. | Envelope do mecanismo real. | Hipótese temporária |
| HT-007 | Módulo Hip Roll | cápsula lateral: `0,036 m` entre centros, raio `0,016 m`, comprimento de colisão `0,068 m` | Volume simples de articulação. | Envelope e auto-interferência. | Hipótese temporária |
| HT-008 | Coxa | `0,125 m` entre quadril e joelho; cápsula de raio `0,025 m`, comprimento de colisão `0,175 m` | Coxa mais larga para acomodar transmissão. | Comprimento, diâmetro e espaço interno. | Hipótese temporária |
| HT-009 | Canela | `0,105 m` entre joelho e tornozelo; cápsula de raio `0,017 m`, comprimento de colisão `0,139 m` | Massa distal e largura menores que a coxa. | Comprimento, resistência e espaço interno. | Hipótese temporária |
| HT-010 | Módulo do tornozelo | cilindro, raio `0,019 m`, altura `0,044 m`; origem `0,105 m` abaixo do joelho | Corpo independente desde a primeira versão. | Atuadores e amplitude mecânica. | Hipótese temporária |
| HT-011 | Pé | box `0,090 × 0,064 × 0,030 m`; centro em `(0,012, 0, -0,023) m` relativo ao tornozelo | Pé pequeno sem aumento artificial para facilitar RL. | Tamanho mínimo estável e fabricação. | Hipótese temporária |
| HT-012 | Piso | plane com `size="2 2 0.05"`; colisão de plane é ilimitada e o tamanho afeta visualização | Área visual suficiente para inspeção atual. | Não se aplica ao robô; revisar para cenários futuros. | Hipótese temporária |

### 4.2 Alturas derivadas e intenção física

| ID | Valor | Como é obtido | Classificação |
| --- | --- | --- | --- |
| VD-001 | Altura do modelo parcial na pose nominal: `0,437125 m` do solo ao topo do tronco. | Altura nominal da raiz `0,292125` + offset do tronco `0,085` + meia altura `0,060`; as solas ficam em `z≈0`. | Valor derivado |
| VD-002 | Altura nominal da pelve: `0,292125 m`. | `0,025 + (0,125+0,105)×cos(5°) + 0,023 + 0,015`; apoia as solas no piso com a pose nominal flexionada. | Valor derivado |

### 4.3 Por que existem `0,303 m` e `0,292125 m`

Os valores pertencem a estados iniciais diferentes:

- o estado cru compilado do MJCF usa `z=0,303 m` e todas as 12 juntas em `0°`.
  A extensão vertical entre raiz e sola é
  `0,025+0,125+0,105+0,023+0,015 = 0,293 m`; portanto cada sola começa
  exatamente `0,010 m` acima do plano;
- `BipedalTwinEnv.reset()` primeiro troca as dez juntas atuadas para a pose
  nominal `(0°, -5°, 10°, 0°, -5°)` por lado. Coxa e canela passam a ter queda
  vertical combinada de `(0,125+0,105)×cos(5°)`, resultando em VD-002. O reset
  então sobrescreve o `z` da raiz e deixa as solas em `≈2,2e-7 m`, diferença
  residual de arredondamento.

A diferença de `0,010875 m` entre as raízes é, portanto, **esperada no
comportamento atual**: o MJCF cru representa pose reta com 10 mm de folga; o
ambiente representa pose nominal flexionada em contato com o piso. Não há conflito
durante um episódio, porque o ambiente sempre sobrescreve o valor cru. Há, porém,
um risco de manutenção: duas referências hardcoded podem divergir se a geometria
ou a pose mudar. Nenhuma delas foi alterada nesta auditoria.

A dimensão pretendida conhecida do hardware é apenas DP-001 (`~0,50 m` final). A
diferença para VD-001 reserva, conceitualmente, partes ainda ausentes, mas não fixa
dimensões de cabeça, pescoço, braços ou carenagem.

## 5. Densidades, massas e inércias

Não há elementos `<inertial>` explícitos. MuJoCo calcula massa e diagonal de
inércia a partir de geometria e `density`. Portanto, as densidades são hipóteses;
massas e inércias abaixo são valores derivados da compilação atual, não previsões
do hardware.

### 5.1 Densidades configuradas

| ID | Geometrias | Densidade (`kg/m³`) | Justificativa | Classificação |
| --- | --- | ---: | --- | --- |
| HT-013 | Default de geom | `500` | Base genérica do modelo; geometrias estruturais atuais sobrescrevem esse valor. | Hipótese temporária |
| HT-014 | Núcleo e box da pelve | `900` | Concentrar massa perto da pelve/COM. | Hipótese temporária |
| HT-015 | Tronco | `600` | Carga superior moderada. | Hipótese temporária |
| HT-016 | Módulos de quadril e coxas | `650` | Representar estrutura e futuros mecanismos proximais. | Hipótese temporária |
| HT-017 | Canelas | `450` | Reduzir massa distal. | Hipótese temporária |
| HT-018 | Tornozelos e pés | `550` | Compromisso provisório entre estrutura e massa distal. | Hipótese temporária |

### 5.2 Valores compilados

Inércias são diagonais principais `(Ixx, Iyy, Izz)` em `kg·m²`. Componentes L/R
são simétricos e o percentual indicado é por componente.

| ID | Corpo | Massa (`kg`) | Parcela | Inércia diagonal (`kg·m²`) | Classificação |
| --- | --- | ---: | ---: | --- | --- |
| VD-003 | Total do robô parcial | `2,675227834` | `100%` | Soma dos corpos móveis | Valor derivado |
| VD-004 | `pelvis_root` | `0,081000000` | `3,028%` | `(3,0375e-5, 2,2950e-5, 4,1175e-5)` | Valor derivado |
| VD-005 | `pelvis` | `0,891000000` | `33,306%` | `(2,591325e-3, 1,084050e-3, 3,304125e-3)` | Valor derivado |
| VD-006 | `trunk` | `0,842400000` | `31,489%` | `(2,197260e-3, 1,579500e-3, 1,755000e-3)` | Valor derivado |
| VD-007 | Hip Yaw L/R, cada | `0,023720281` | `0,887%` | `(4,008728e-6, 4,008728e-6, 5,740308e-6)` | Valor derivado |
| VD-008 | Hip Roll L/R, cada | `0,029971632` | `1,120%` | `(1,040113e-5, 1,040113e-5, 3,550872e-6)` | Valor derivado |
| VD-009 | Coxa L/R, cada | `0,202076402` | `7,554%` | `(4,593250e-4, 4,593250e-4, 6,048998e-5)` | Valor derivado |
| VD-010 | Canela L/R, cada | `0,052160020` | `1,950%` | `(7,530767e-5, 7,530767e-5, 7,269486e-6)` | Valor derivado |
| VD-011 | Tornozelo L/R, cada | `0,027445582` | `1,026%` | `(6,904851e-6, 6,904851e-6, 4,953928e-6)` | Valor derivado |
| VD-012 | Pé L/R, cada | `0,095040000` | `3,553%` | `(3,956832e-5, 7,128000e-5, 9,659232e-5)` | Valor derivado |

A pelve estrutural (`pelvis_root + pelvis`) representa aproximadamente `36,334%`
da massa, o tronco `31,489%`, as duas coxas `15,108%`, as duas canelas `3,900%`
e os dois pés `7,106%`; módulos de quadril e tornozelo completam o total. Essa
distribuição é derivada das hipóteses HT-002–HT-018.

## 6. Juntas, DOFs e limites

Os ranges são hipóteses iniciais sem medição de servo, batente, cabeamento ou
auto-interferência do hardware. Os `ctrlrange` dos dez atuadores coincidem com os
ranges abaixo (em radianos no modelo compilado).

| ID | Juntas L/R | Eixo | Range atual | Atuador | Bloqueio nesta fase | Justificativa | Classificação |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HT-019 | Hip Yaw | `(0,0,1)` | `[-30°, 30°]` | Não | Equality constraint temporária em `q=0` | Cumprir o bloqueio inicial planejado sem remover as hinges nem alterar a interface de 10 ações. | Hipótese temporária |
| HT-020 | Hip Roll | `(1,0,0)` | `[-35°, 35°]` | Sim, posição | Não | Inclinação frontal suficiente para experimentação inicial. | Hipótese temporária |
| HT-021 | Hip Pitch | `(0,1,0)` | `[-90°, 45°]` | Sim, posição | Não | Flexão ampla e extensão limitada provisoriamente. | Hipótese temporária |
| HT-022 | Knee Pitch | `(0,1,0)` | `[0°, 130°]` | Sim, posição | Não | Evitar hiperextensão e permitir flexão. | Hipótese temporária |
| HT-023 | Ankle Roll | `(1,0,0)` | `[-30°, 30°]` | Sim, posição | Não | Ajuste lateral inicial. | Hipótese temporária |
| HT-024 | Ankle Pitch | `(0,1,0)` | `[-45°, 45°]` | Sim, posição | Não | Ajuste sagital inicial. | Hipótese temporária |

| ID | Contagem compilada | Classificação |
| --- | --- | --- |
| VD-013 | `13` joints no total: `1` freejoint + `12` articulares; `nq=19`, `nv=18`, `nu=10`. | Valor derivado |

Importante: “sem atuação” não equivale por si só a “bloqueada”. No MJCF atual, o
bloqueio é fornecido separadamente pelas equalities `left_hip_yaw_lock` e
`right_hip_yaw_lock`, ambas com alvo polinomial constante em `0 rad`.

### 6.1 Alternativas para Hip Yaw nesta fase

| Alternativa | Vantagens | Custos e riscos |
| --- | --- | --- |
| Manter passivo como hoje | Nenhuma mudança no MJCF; preserva os 12 DOFs e permite observar a dinâmica livre. | Contraria a intenção original de bloqueio; yaw pode derivar sem comando. As juntas não entram na `nominal_pose`, nem nas parcelas articulares de pose e velocidade, porque estas cobrem somente os dez DOFs atuados. |
| Bloquear fisicamente na simulação | Representa uma conexão rígida e elimina deriva de yaw. | Remover/weldar o DOF muda a árvore e as contagens `nq/nv`, contrariando o requisito de manter Hip Yaw estruturalmente presente e exigindo reestruturação para liberá-lo depois. |
| Manter a hinge e aplicar bloqueio temporário equivalente | Preserva nomes, árvore, range futuro e interface externa de dez ações; uma equality constraint em `q=0` pode ser removida numa fase posterior. | Introduz forças de restrição e dependência do solver; exige parâmetros e testes próprios. Um servo interno de posição seria outra variante, mas aumentaria `nu` e poderia interferir na penalidade de esforço. |

**Decisão aprovada e implementada:** foram mantidas as duas hinges e adicionadas
duas restrições de igualdade temporárias em `q=0`. O modelo continua com 12 juntas
articulares, `nq=19`, `nv=18`, `nu=10` e action space de 10 ações. O bloqueio pode
ser retirado futuramente sem reconstruir a árvore cinemática.

**HT-067 — hipótese temporária de validação:** considerar o bloqueio efetivo se o
valor absoluto de cada Hip Yaw permanecer abaixo de `0,25°` em rollouts com as
perturbações padrão. Esse limite é pequeno frente ao range estrutural de `±30°` e
mantém margem sobre a deriva numérica observada, sem ser apresentado como
tolerância mecânica do hardware.

## 7. Pose nominal e interface de ação

Todos os ângulos nominais são hipóteses temporárias. Valores iguais nos lados L/R
mantêm simetria nominal.

| ID | Junta atuada | Nominal | Justificativa | Classificação |
| --- | --- | ---: | --- | --- |
| HT-025 | Left Hip Roll | `0°` | Pelve nivelada no plano frontal. | Hipótese temporária |
| HT-026 | Left Hip Pitch | `-5°` | Pequena flexão sagital. | Hipótese temporária |
| HT-027 | Left Knee Pitch | `10°` | Evitar joelho totalmente estendido. | Hipótese temporária |
| HT-028 | Left Ankle Roll | `0°` | Sola nivelada lateralmente. | Hipótese temporária |
| HT-029 | Left Ankle Pitch | `-5°` | Compensar hip+knee e manter o pé horizontal. | Hipótese temporária |
| HT-030 | Right Hip Roll | `0°` | Pelve nivelada no plano frontal. | Hipótese temporária |
| HT-031 | Right Hip Pitch | `-5°` | Pequena flexão sagital. | Hipótese temporária |
| HT-032 | Right Knee Pitch | `10°` | Evitar joelho totalmente estendido. | Hipótese temporária |
| HT-033 | Right Ankle Roll | `0°` | Sola nivelada lateralmente. | Hipótese temporária |
| HT-034 | Right Ankle Pitch | `-5°` | Compensar hip+knee e manter o pé horizontal. | Hipótese temporária |
| HT-035 | Orientação nominal da raiz | Quaternion `(1,0,0,0)`, roll/pitch/yaw `0°` | Tronco vertical e pés paralelos ao chão. | Hipótese temporária |

| ID | Resultado nominal | Classificação |
| --- | --- | --- |
| VD-014 | Tronco com inclinação `0°` e solas em `z≈0` quando a pose nominal usa VD-002. | Valor derivado |
| VD-016 | Escalas negativa/positiva da ação são derivadas da distância entre pose nominal e a interseção `joint range ∩ ctrlrange`. | Valor derivado |

`action=0` seleciona exatamente os dez ângulos acima; não seleciona o ponto médio
arbitrário dos ranges. Para cada componente normalizado `a`:

```text
a < 0: target = nominal + a × (nominal - limite_inferior)
a > 0: target = nominal + a × (limite_superior - nominal)
a = 0: target = nominal
```

O target final é limitado à interseção entre range físico e `ctrlrange`.

## 8. Atuadores e dinâmica articular

| ID | Parâmetro | Valor atual | Origem/justificativa | Validar no hardware | Classificação |
| --- | --- | --- | --- | --- | --- |
| HT-036 | `kp` de Hip Roll, Hip Pitch e Knee Pitch | `25` | Ganho inicial de atuador de posição. | Resposta do servo/transmissão. | Hipótese temporária |
| HT-037 | `kp` de Ankle Roll/Pitch | `20` | Ganho um pouco menor nos atuadores distais. | Resposta do servo/transmissão. | Hipótese temporária |
| HT-038 | Damping de cada hinge | `0,2` | Amortecimento numérico/mecânico inicial. | Atrito e damping reais por junta. | Hipótese temporária |
| HT-039 | Armature de cada hinge | `0,005` | Inércia refletida provisória para estabilidade dinâmica. | Rotor, redutor e transmissão reais. | Hipótese temporária |

O tipo MuJoCo atual é `<position>`. Há dez atuadores e os `ctrlrange` são:

- Hip Roll: `[-0,610865, 0,610865] rad` (`±35°`);
- Hip Pitch: `[-1,570796, 0,785398] rad` (`-90°` a `45°`);
- Knee Pitch: `[0, 2,268928] rad` (`0°` a `130°`);
- Ankle Roll: `[-0,523599, 0,523599] rad` (`±30°`);
- Ankle Pitch: `[-0,785398, 0,785398] rad` (`±45°`).

Esses ranges compartilham as classificações HT-020–HT-024. Hip Yaw não possui
atuador. `data.ctrl` é target de posição; esforço é avaliado por
`data.actuator_force`, não pelo valor do target.

## 9. Recompensa da Etapa 4

Todos os itens desta seção são hipóteses temporárias e vivem em
`PostureRewardConfig`.

| ID | Componente/peso | Valor | Intenção | Classificação |
| --- | --- | ---: | --- | --- |
| HT-040 | `upright_weight` | `1,0` | Favorecer alinhamento do tronco com a vertical. | Hipótese temporária |
| HT-041 | `height_weight` | `0,5` | Favorecer altura próxima da referência nominal. | Hipótese temporária |
| HT-042 | `pose_weight` | `0,25` | Preferir suavemente a pose nominal sem obrigá-la. | Hipótese temporária |
| HT-043 | `velocity_weight` | `0,1` | Desencorajar movimento rápido/violento. | Hipótese temporária |
| HT-044 | `effort_weight` | `0,01` | Desencorajar esforço físico excessivo. | Hipótese temporária |
| HT-045 | Escala de inclinação | `30°` | Queda progressiva de `upright_reward` com inclinação perceptível. | Hipótese temporária |
| HT-046 | Escala de altura | `25%` de VD-002 | Rejeitar fortemente pelve próxima do piso. | Hipótese temporária |
| HT-047 | Escala da pose | erro RMS de `20°` | Tolerar pequenos desvios articulares. | Hipótese temporária |
| HT-048 | Escala de velocidade articular | `5 rad/s` | Normalização de velocidade rápida. | Hipótese temporária |
| HT-049 | Escala de velocidade angular do tronco | `5 rad/s` | Tornar rotação instável numericamente visível. | Hipótese temporária |
| HT-050 | Escala de força do atuador | `20 N·m` | Normalizar momento dos atuadores hinge. | Hipótese temporária |
| HT-051 | Penalidade sentinela para velocidade/força não finita | `1,0e6` | Manter recompensa finita enquanto o estado é marcado como queda. | Hipótese temporária |

As recompensas positivas são gaussianas: alinhamento usa inclinação, altura usa o
erro relativo a VD-002 e pose usa erro RMS das dez juntas. `velocity_penalty` soma
médias quadráticas normalizadas das velocidades articulares e angular do tronco;
`effort_penalty` usa a média quadrática de `actuator_force/20`. A composição é:

```text
reward_total =
    1.0 × upright_reward
  + 0.5 × height_reward
  + 0.25 × pose_reward
  - 0.1 × velocity_penalty
  - 0.01 × effort_penalty
```

## 10. Queda e duração do episódio

| ID | Critério/parâmetro | Valor | Justificativa | Classificação |
| --- | --- | --- | --- | --- |
| HT-052 | Altura mínima da pelve | `50%` de VD-002, atualmente `0,1460625 m` | Estado claramente próximo do chão. | Hipótese temporária |
| HT-053 | Inclinação máxima do tronco | `60°` | Limite inicial de postura claramente inválida. | Hipótese temporária |
| HT-065 | Limite padrão do episódio | `1000` passos | Horizonte configurável inicial, ainda não associado a uma tarefa treinada. | Hipótese temporária |
| VD-015 | Duração simulada do limite padrão | `2,0 s` | `1000 × 0,002 s`. | Valor derivado |

Qualquer `NaN` ou `Inf` em `qpos`, `qvel`, posição da pelve ou matriz do tronco
marca `is_fallen=True`. Queda física/numericamente inválida retorna
`terminated=True`. O limite de passos, somente quando não há queda, retorna
`truncated=True`.

## 11. Reset perturbado da Etapa 5

Todas as categorias podem ser desligadas individualmente ou pelo campo global
`enabled`. Amplitudes zero também restauram a condição nominal. A distribuição é
uniforme, independente por componente, centrada em zero e amostrada por
`self.np_random`; repetir a mesma seed reproduz o estado, enquanto `reset()` sem
seed continua a sequência do RNG.

| ID | Perturbação/configuração padrão | Valor | Justificativa | Classificação |
| --- | --- | --- | --- | --- |
| HT-054 | Posição das dez juntas atuadas | `±2°` | Pequena frente à menor margem nominal, de `10°`. | Hipótese temporária |
| HT-055 | Roll inicial | `±1°` | Imperfeição pequena mantendo verticalidade. | Hipótese temporária |
| HT-056 | Pitch inicial | `±1°` | Imperfeição pequena mantendo verticalidade. | Hipótese temporária |
| HT-057 | Velocidade articular inicial | `±0,1 rad/s` | Movimento inicial pequeno, não um empurrão. | Hipótese temporária |
| HT-058 | Velocidade angular inicial de roll/pitch | `±0,05 rad/s` | Rotação inicial pequena, não um empurrão. | Hipótese temporária |
| HT-059 | Perturbação de yaw | `0°` | Mudança de direção está fora do escopo atual. | Hipótese temporária |
| HT-060 | Velocidade angular de yaw | `0 rad/s` | Mudança de direção está fora do escopo atual. | Hipótese temporária |
| HT-061 | Velocidade linear da raiz | `(0,0,0) m/s` | Evitar lançar o robô no reset. | Hipótese temporária |
| HT-062 | Distribuição | Uniforme e independente em `[-amplitude,+amplitude]` | Simetria estatística simples entre lados. | Hipótese temporária |
| HT-064 | Penetração máxima contra o solo aceita na validação | `0,01 m` | Limite conservador de teste frente à escala do robô; não altera contatos. | Hipótese temporária |

Antes de amostrar, a amplitude articular é validada contra a menor margem da pose
nominal aos ranges. A combinação máxima de roll/pitch é validada contra HT-053.
Depois da amostragem, o quaternion é normalizado, `mujoco.mj_forward()` é chamado e
estado não finito ou já caído gera erro explícito, sem correção arbitrária.

## 12. Parâmetros físicos e numéricos do MuJoCo

| ID | Parâmetro | Valor explícito | Origem/justificativa | Classificação |
| --- | --- | --- | --- | --- |
| HT-063 | `timestep` | `0,002 s` (`500 Hz`) | Resolução temporal inicial para o robô pequeno. | Hipótese temporária |
| DP-009 | `gravity` | `(0,0,-9,81) m/s²` | Operação terrestre nominal. | Decisão de projeto |
| DP-010 | `integrator` | `implicitfast` | Integração atual. | Decisão de projeto |
| HT-038 | `joint damping` | `0,2` | Herdado por hinges. | Hipótese temporária |
| HT-039 | `joint armature` | `0,005` | Herdado por hinges. | Hipótese temporária |
| HT-013 | `geom density` default | `500 kg/m³` | Default explícito; overrides estão na seção 5. | Hipótese temporária |
| HT-066 | `geom friction` | `(0,8, 0,1, 0,1)` | Atrito deslizante, torsional e de rolamento ainda não calibrados. | Hipótese temporária |
| DP-011 | `compiler` | `angle="degree"`, `autolimits="true"` | Legibilidade e ranges ativos. | Decisão de projeto |

Não há `solref`, `solimp`, cone, solver, iterações ou tolerâncias explicitamente
alterados: os defaults da versão instalada do MuJoCo são usados. Esses defaults não
são copiados aqui para evitar tratá-los como decisões do projeto.

## 13. Decisões em aberto

| ID | Decisão deliberadamente pendente | O que precisa resolvê-la | Classificação |
| --- | --- | --- | --- |
| DA-001 | Servo/motor físico por junta. | Seleção e ensaio de bancada. | Decisão em aberto |
| DA-002 | Torque nominal, pico e necessário por junta. | Dimensionamento dinâmico e margem de segurança. | Decisão em aberto |
| DA-003 | Velocidade necessária por junta. | Trajetórias e ensaio do primeiro protótipo. | Decisão em aberto |
| DA-004 | Relação e mecanismo do redutor. | Seleção de motor e transmissão. | Decisão em aberto |
| DA-005 | Resolução, backlash e precisão. | Encoder, redutor e controle físico. | Decisão em aberto |
| DA-006 | Comportamento térmico/elétrico dos atuadores. | Curvas e ensaios sob carga. | Decisão em aberto |
| DA-007 | Interface e uso futuro de controle direto por torque. | Hardware e objetivos de controle posteriores. | Decisão em aberto |
| DA-008 | Massas reais de estrutura e mecanismos. | CAD, materiais e pesagem. | Decisão em aberto |
| DA-009 | Material estrutural. | Requisitos de rigidez, massa e fabricação. | Decisão em aberto |
| DA-010 | Espessura das peças impressas. | CAD, material e ensaios mecânicos. | Decisão em aberto |
| DA-011 | Tipo definitivo de transmissão. | Torque, velocidade, backlash e integração. | Decisão em aberto |
| DA-012 | Bateria: química, capacidade, massa e posição. | Orçamento elétrico e autonomia. | Decisão em aberto |
| DA-013 | Computador embarcado. | Carga computacional, energia e interfaces. | Decisão em aberto |
| DA-014 | Controladores, cabeamento e sua distribuição de massa. | Arquitetura elétrica e layout. | Decisão em aberto |
| DA-015 | Sensores e respectivas posições/ruídos. | Estratégia de controle e hardware. | Decisão em aberto |
| DA-016 | Braços, cabeça e pescoço, incluindo massas e DOFs. | Escopo mecânico posterior. | Decisão em aberto |
| DA-017 | Tamanho definitivo dos pés. | Protótipo físico e estabilidade sem inflação para RL. | Decisão em aberto |
| DA-018 | Dimensões finais da pelve. | Componentes internos e ergonomia. | Decisão em aberto |
| DA-019 | Demais dimensões finais e correspondência CAD–MJCF. | Projeto mecânico detalhado. | Decisão em aberto |
| DA-020 | Método futuro de sim-to-real. | Resultados de simulação e medições reais. | Decisão em aberto |
| DA-021 | Necessidade e parâmetros de domain randomization. | Caracterização do gap sim-to-real; não implementado. | Decisão em aberto |
| DA-022 | Pesos e escalas definitivos da reward. | Experimentos de RL e análise de reward hacking. | Decisão em aberto |
| DA-023 | Limites definitivos de queda. | Segurança, hardware e dados experimentais. | Decisão em aberto |
| DA-024 | Ranges físicos finais e eventual bloqueio mecânico de Hip Yaw. | Batentes, transmissão e cabeamento. | Decisão em aberto |
| DA-025 | Fricção, contatos e parâmetros de solver calibrados. | Medições de materiais e contatos reais. | Decisão em aberto |
| DA-026 | Damping, armature e inércias equivalentes reais. | Identificação de sistema. | Decisão em aberto |
| DA-027 | Horizonte definitivo dos episódios e protocolo de treino. | Definição das tarefas futuras. | Decisão em aberto |
| DA-028 | Carenagem removível: material, dimensões e massa. | Design industrial e integração mecânica. | Decisão em aberto |

Também permanecem fora do modelo atual as massas de servos, bateria, computador,
controladores, cabeamento, carenagem, braços e cabeça. Nada na tabela de massas
simulada deve ser usado como previsão desses itens.

## 14. Rastreabilidade por grupo

| Grupo | Implementação executável | Registro/validação |
| --- | --- | --- |
| Geometria, densidade, massas, juntas, ranges, dinâmica e contatos | `assets/mujoco/models/digital_twin_v0_1.xml` | Este documento; `tests/test_digital_twin_model.py` |
| Pose nominal, altura inicial e mapeamento da ação | `src/companion_robot/envs/bipedal_twin_env.py` | Este documento; `docs/nominal-pose-v0.1.md`; testes do ambiente |
| Pesos e escalas da recompensa | `PostureRewardConfig` no ambiente | Este documento; `tests/test_bipedal_posture_evaluation.py` |
| Queda e término | `FallDetectionConfig` e `BipedalTwinEnv.step()` | Este documento; testes de postura |
| Reset perturbado | `ResetPerturbationConfig` e `BipedalTwinEnv.reset()` | Este documento; `tests/test_bipedal_reset_perturbation.py` |
| Interface Gymnasium e horizonte | `BipedalTwinEnv` | Testes de contrato e `check_env` |
| Requisitos do robô físico | Não é implementação executável | `docs/design-spec-v0.1.md` |

## 15. Auditoria das inconsistências e estado atual

1. **Resolvida:** a visão geral, a estrutura de arquivos, o risco e a conclusão de
   `docs/Implementação.md` agora reconhecem o bípede e as Etapas 1–6 concluídas.
2. **Resolvida:** `docs/nominal-pose-v0.1.md` classifica `0,292125 m` como VD-002,
   derivado de hipóteses de pose/geometria.
3. **Resolvida para esta fase:** Hip Yaw preserva as hinges e o range estrutural
   `±30°`, mas duas equality constraints temporárias mantêm L/R em `q=0`. DA-024
   permanece aberta apenas para ranges finais e eventual solução mecânica.
4. **Investigada e esperada:** `0,303 m` pertence ao estado cru reto com 10 mm de
   folga; `0,292125 m` pertence à pose nominal flexionada apoiada. A seção 4.3
   contém a derivação e registra o risco de manutenção das duas referências.
5. **Resolvida:** o roadmap de `docs/design-spec-v0.1.md` marca o primeiro bípede
   como concluído, sem marcar RL ou etapas posteriores.
6. **Resolvida:** critérios e descrições em `docs/Implementação.md` usam os nomes
   literais das cadeias atuais, em vez de `left_leg`/`right_leg`.

## 16. Verificação automática e limites do registro

Os testes verificam invariantes estruturais, não uma cópia textual de todo o XML:

- 12 juntas articulares e um freejoint;
- 10 atuadores e ausência de Hip Yaw entre eles;
- duas equalities de junta ativas com alvo `q=0` para Hip Yaw L/R;
- ordem, eixos e árvore cinemática;
- pose nominal dentro dos ranges;
- dimensões relevantes positivas;
- massas e inércias finitas e positivas;
- contratos Gymnasium, rewards, quedas e reset perturbado.

Os valores derivados deste documento foram extraídos do modelo compilado atual.
Se geometria ou densidade mudar, as massas e a altura derivada devem ser novamente
extraídas e este documento atualizado. Essa atualização não é automática nesta
versão.
