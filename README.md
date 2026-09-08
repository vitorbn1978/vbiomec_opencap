# VBiomec OpenCap

Ferramenta web desenvolvida em **Python** e **Streamlit** para processamento, visualização e emissão de laudos de biomecânica da corrida utilizando dados cinemáticos e cinéticos (como OpenCap e Sport2D).

## Funcionalidades
* **Análise Espaço-Temporal**: Métricas detalhadas de tempo de passada, comprimento e cadência.
* **Cinemática Articular (IK)**: Avaliação de ângulos de quadril, joelho e tornozelo ao longo do ciclo da passada.
* **Atuação Muscular e Potência**: Curvas de força (BW) e potência muscular (W/kg), destacando fases de frenagem excêntrica e propulsão concêntrica de grupos musculares-chave (isquiotibiais, quadríceps, plantiflexores e glúteos).
* **Função de Simetria**: Avaliação quantitativa das assimetrias entre os membros inferior direito e esquerdo.
* **Geração de Laudos em PDF**: Exportação de relatórios profissionais completos prontos para uso clínico ou esportivo.

## Requisitos
As principais bibliotecas utilizadas no projeto estão listadas no arquivo `requirements.txt`:
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `reportlab`
- `opencv-python`
- `scipy`

## Como Executar Localmente

1. Clone o repositório:
   ```bash
   git clone [https://github.com/vitorbn1978/vbiomec_opencap.git](https://github.com/vitorbn1978/vbiomec_opencap.git)
   cd vbiomec_opencap
## Link
https://vbiomecrun.streamlit.app/
