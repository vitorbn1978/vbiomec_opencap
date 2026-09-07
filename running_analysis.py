import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
import io

# ReportLab para geração de PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- PARSER E FREQUÊNCIA DE AMOSTRAGEM ---
def carregar_arquivo_opensim(file):
    content = file.getvalue().decode('utf-8')
    linhas = content.splitlines()
    
    header_end = 0
    in_degrees = True
    for idx, linha in enumerate(linhas):
        if 'inDegrees=no' in linha:
            in_degrees = False
        if 'endheader' in linha.lower():
            header_end = idx + 1
            break
            
    file.seek(0)
    df = pd.read_csv(file, skiprows=header_end, sep=r'\s+')
    return df, in_degrees

def extrair_freq_amostragem(df):
    tempo = df['time'].values
    dt = np.diff(tempo)
    dt_medio = np.mean(dt)
    fs = 1.0 / dt_medio if dt_medio > 0 else 100.0
    return fs, dt_medio

def filtrar_sinal(sinal, fs=100.0, fc=8.0):
    if len(sinal) < 15:
        return sinal
    nyq = 0.5 * fs
    normal_cutoff = fc / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, sinal)

# --- MÉTODOS DE LEITCH ET AL. (2011) ---
def extrair_heel_strikes_leitch(df_pos, sufixo_lado='r'):
    col_y = f'calcn_{sufixo_lado}_Y'
    if col_y not in df_pos.columns:
        return []
        
    fs, _ = extrair_freq_amostragem(df_pos)
    calcn_y = filtrar_sinal(df_pos[col_y].values, fs=fs, fc=8.0)
    
    v_y = np.gradient(calcn_y) * fs
    
    min_y = np.min(calcn_y)
    max_y = np.max(calcn_y)
    threshold_y = min_y + 0.35 * (max_y - min_y)
    min_dist_frames = max(1, int(0.25 * fs))
    
    peaks, _ = find_peaks(-calcn_y, distance=min_dist_frames, height=-threshold_y, prominence=0.015)
    
    foot_strikes = []
    for p in peaks:
        idx_zero_cross = p
        janela_busca = range(max(1, p - 10), min(len(v_y) - 1, p + 5))
        
        for i in janela_busca:
            if v_y[i] <= 0 and v_y[i + 1] > 0:
                idx_zero_cross = i
                break
                
        foot_strikes.append(idx_zero_cross)
        
    return np.array(foot_strikes)

def calcular_espaco_temporais_membro(df_pos, hs_indices, vel_esteira_kmh=10.0):
    if len(hs_indices) < 2:
        return 0.0, 0.0, 0.0, 0.0
        
    tempo = df_pos['time'].values
    duracoes = [tempo[hs_indices[i+1]] - tempo[hs_indices[i]] for i in range(len(hs_indices)-1)]
    t_passada_medio = np.mean(duracoes)
    
    vel_ms = vel_esteira_kmh / 3.6
    comp_passada_medio = vel_ms * t_passada_medio
    cadencia_media = (60.0 / t_passada_medio) * 2.0 if t_passada_medio > 0 else 0.0
    
    return t_passada_medio, comp_passada_medio, vel_ms, cadencia_media

# --- ENSEMBLE AVERAGE & SIMETRIA ---
def extrair_ensemble_average(df, coluna_tempo, dict_colunas_ou_grupos, hs_indices, peso_corporal=1.0):
    if len(hs_indices) < 2:
        return None
        
    percentual_ciclo = np.linspace(0, 100, 101)
    tempo = df[coluna_tempo].values
    resultados = {'% Ciclo': percentual_ciclo}
    
    for nome_var, colunas in dict_colunas_ou_grupos.items():
        if isinstance(colunas, str):
            colunas = [colunas]
            
        cols_existentes = [c for c in colunas if c in df.columns]
        if cols_existentes:
            soma_bruta = df[cols_existentes].sum(axis=1).values / peso_corporal
            ciclos_matriz = []
            
            for i in range(len(hs_indices) - 1):
                idx_i = hs_indices[i]
                idx_f = hs_indices[i + 1]
                
                t_ciclo = tempo[idx_i:idx_f]
                t_norm = np.linspace(t_ciclo[0], t_ciclo[-1], 101)
                sinal_ciclo = soma_bruta[idx_i:idx_f]
                
                sinal_interp = np.interp(t_norm, t_ciclo, sinal_ciclo)
                ciclos_matriz.append(sinal_interp)
                
            ciclos_matriz = np.array(ciclos_matriz)
            resultados[f'{nome_var}_mean'] = np.mean(ciclos_matriz, axis=0)
            resultados[f'{nome_var}_std'] = np.std(ciclos_matriz, axis=0)
            resultados[f'{nome_var}_cycles'] = ciclos_matriz
            
    return resultados

def symmetry_plot(xR, xL):
    xR = np.array(xR)
    xL = np.array(xL)
    
    avg_rom = ((np.max(xR) - np.min(xR)) + (np.max(xL) - np.min(xL))) / 2.0
    if avg_rom == 0:
        avg_rom = 1e-6
        
    diff_pct = ((xR - xL) / avg_rom) * 100.0
    time_idx = np.linspace(0, 100, len(diff_pct))
    smoothed = savgol_filter(diff_pct, window_length=15, polyorder=2)
    
    return pd.DataFrame({
        'time': time_idx,
        'score': diff_pct,
        'score_smoothed': smoothed
    })

# --- GERADOR DE PDF ---
def gerar_pdf_laudo(nome_atleta, vel_esteira_kmh, observacoes, dados_et, figuras_selecionadas):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, leading=20, alignment=1, textColor=colors.HexColor('#1E3A8A'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=11, leading=15, textColor=colors.HexColor('#1E3A8A'))
    normal_style = styles['Normal']

    story.append(Paragraph("LAUDO DE ANÁLISE BIOMECÂNICA DA CORRIDA", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Atleta / Avaliado:</b> {nome_atleta}", normal_style))
    story.append(Paragraph(f"<b>Data da Análise:</b> {pd.Timestamp.now().strftime('%d/%m/%Y')} | <b>Velocidade Controlada:</b> {vel_esteira_kmh:.1f} km/h", normal_style))
    story.append(Spacer(1, 12))

    if dados_et:
        story.append(Paragraph("1. Parâmetros Espaço-Temporais Média Ensemble", heading_style))
        story.append(Spacer(1, 4))
        
        tabela_dados = [
            ["Parâmetro", "Lado Direito (R)", "Lado Esquerdo (L)"],
            ["Tempo Médio da Passada", f"{dados_et['t_r']:.2f} s", f"{dados_et['t_l']:.2f} s"],
            ["Comprimento Médio da Passada", f"{dados_et['comp_r']:.2f} m", f"{dados_et['comp_l']:.2f} m"],
            ["Velocidade Média", f"{dados_et['v_r']:.2f} m/s ({dados_et['v_r']*3.6:.1f} km/h)", f"{dados_et['v_l']:.2f} m/s ({dados_et['v_l']*3.6:.1f} km/h)"],
            ["Cadência Média Estimada", f"{dados_et['cad_r']:.1f} ppm", f"{dados_et['cad_l']:.1f} ppm"],
        ]
        
        t = Table(tabela_dados, colWidths=[180, 160, 160])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1F2937')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#D1D5DB')),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    if observacoes:
        story.append(Paragraph("2. Observações Clínicas e Diagnóstico Biomecânico", heading_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(observacoes.replace('\n', '<br/>'), normal_style))
        story.append(Spacer(1, 12))

    if figuras_selecionadas:
        story.append(Paragraph("3. Gráficos Ensemble Average Selecionados", heading_style))
        story.append(Spacer(1, 8))
        
        for titulo_grafico, fig in figuras_selecionadas:
            story.append(Paragraph(f"<b>{titulo_grafico}</b>", normal_style))
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            
            story.append(RLImage(img_buffer, width=460, height=210))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer


# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Running Biomechanics & Power - OpenSim", layout="wide")
st.title("🏃 Running Biomechanics: Gait, Force, Power & Symmetry")

st.sidebar.header("1. Upload dos Arquivos (.sto / .mot)")
file_positions = st.sidebar.file_uploader("1. Posições / Corpos (Kinematics)", type=['sto', 'mot', 'txt'])
file_ik = st.sidebar.file_uploader("2. Ângulos Articulares (IK)", type=['sto', 'mot', 'txt'])
file_muscles = st.sidebar.file_uploader("3. Ativação / Força Muscular", type=['sto', 'mot', 'txt'])
file_power = st.sidebar.file_uploader("4. Potência Muscular (Power)", type=['sto', 'mot', 'txt'])

st.sidebar.header("2. Configuração do Protocolo")
vel_esteira_kmh = st.sidebar.number_input("Velocidade da Esteira (km/h):", min_value=1.0, max_value=30.0, value=10.0, step=0.5)
peso_corporal = st.sidebar.number_input("Peso Corporal (kg):", min_value=1.0, max_value=200.0, value=66.0, step=1.0)

dados_et = {}
hs_r = []
hs_l = []

# --- 1. ESPAÇO-TEMPORAIS ---
st.header("1. Detecção do Ciclo de Marcha (Método Leitch et al. 2011)")

if file_positions is not None:
    df_pos, _ = carregar_arquivo_opensim(file_positions)
    fs, dt_medio = extrair_freq_amostragem(df_pos)
    
    hs_r = extrair_heel_strikes_leitch(df_pos, 'r')
    hs_l = extrair_heel_strikes_leitch(df_pos, 'l')
    
    n_ciclos_r = len(hs_r) - 1 if len(hs_r) >= 2 else 0
    n_ciclos_l = len(hs_l) - 1 if len(hs_l) >= 2 else 0
    
    st.caption(f"⚙️ **Amostragem:** `{fs:.1f} Hz` | **Massa:** `{peso_corporal:.1f} kg` | **Ciclos Processados:** `R = {n_ciclos_r}` | `L = {n_ciclos_l}`")
    
    if len(hs_r) >= 2 and len(hs_l) >= 2:
        t_r, comp_r, v_r, cad_r = calcular_espaco_temporais_membro(df_pos, hs_r, vel_esteira_kmh)
        t_l, comp_l, v_l, cad_l = calcular_espaco_temporais_membro(df_pos, hs_l, vel_esteira_kmh)
        
        dados_et = {
            't_r': t_r, 'comp_r': comp_r, 'v_r': v_r, 'cad_r': cad_r,
            't_l': t_l, 'comp_l': comp_l, 'v_l': v_l, 'cad_l': cad_l
        }
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Passada Média Direita (s)", f"{t_r:.2f}s")
        col2.metric("Passada Média Esquerda (s)", f"{t_l:.2f}s")
        col3.metric("Comprimento Médio R (m)", f"{comp_r:.2f}m")
        col4.metric("Comprimento Médio L (m)", f"{comp_l:.2f}m")
    else:
        st.warning("Não foram detectados ciclos completos válidos para ambos os lados.")
else:
    st.info("Aguardando upload do arquivo de Posições.")

dict_figuras_laudo = {}

# --- 2. CINEMÁTICA ENSEMBLE AVERAGE ---
st.header("2. Cinemática Articular: Ensemble Average (Média ± SD)")

if file_ik is not None and len(hs_r) >= 2 and len(hs_l) >= 2:
    df_ik, in_degrees = carregar_arquivo_opensim(file_ik)
    colunas_ik = [c for c in df_ik.columns if c != 'time']
    
    if not in_degrees:
        df_ik[colunas_ik] = df_ik[colunas_ik] * (180.0 / np.pi)
        
    pares_ik = sorted(list(set([c[:-2] for c in colunas_ik if c.endswith('_r') or c.endswith('_l')])))
    
    for art in pares_ik:
        col_r = f"{art}_r"
        col_l = f"{art}_l"
        if col_r in df_ik.columns and col_l in df_ik.columns:
            ens_r = extrair_ensemble_average(df_ik, 'time', {'Articulacao': col_r}, hs_r)
            ens_l = extrair_ensemble_average(df_ik, 'time', {'Articulacao': col_l}, hs_l)
            
            m_r, std_r = ens_r['Articulacao_mean'], ens_r['Articulacao_std']
            m_l, std_l = ens_l['Articulacao_mean'], ens_l['Articulacao_std']
            perc = ens_r['% Ciclo']
            sp = symmetry_plot(m_r, m_l)
            
            fig_ik, ax_ik = plt.subplots(figsize=(6, 3.5))
            ax_ik.plot(perc, m_r, label='Direito (Média)', color='#02579B', linewidth=2)
            ax_ik.fill_between(perc, m_r - std_r, m_r + std_r, color='#02579B', alpha=0.2)
            
            ax_ik.plot(perc, m_l, label='Esquerdo (Média)', color='#ff9933', linewidth=2)
            ax_ik.fill_between(perc, m_l - std_l, m_l + std_l, color='#ff9933', alpha=0.2)
            
            ax_ik.set_xlabel("% do Ciclo de Marcha")
            ax_ik.set_ylabel("Ângulo (°)")
            ax_ik.legend()
            ax_ik.grid(True, linestyle='--', alpha=0.5)
            plt.close(fig_ik)
            dict_figuras_laudo[f"Ensemble Ângulos ({art})"] = fig_ik
            
            fig_sym, ax_sym = plt.subplots(figsize=(6, 3.5))
            ax_sym.scatter(sp['time'], sp['score'], color='gray', alpha=0.3, s=15, label='Bruto')
            ax_sym.plot(sp['time'], sp['score_smoothed'], color='blue', linewidth=2.5, label='Ajustado')
            ax_sym.axvline(x=40, color='gray', linestyle='--', label='Off-foot (40%)')
            ax_sym.set_xlabel("Gait Cycle (%)")
            ax_sym.set_ylabel("Symmetry Function (%)")
            ax_sym.legend()
            ax_sym.grid(True, linestyle='--', alpha=0.5)
            plt.close(fig_sym)
            dict_figuras_laudo[f"Função de Simetria ({art})"] = fig_sym

    default_ik = [a for a in ['hip_flexion', 'knee_angle', 'ankle_angle'] if a in pares_ik]
    art_selecionada = st.selectbox("Selecione a articulação para visualizar na tela:", pares_ik, index=pares_ik.index(default_ik[0]) if default_ik else 0)
    
    if f"Ensemble Ângulos ({art_selecionada})" in dict_figuras_laudo:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.pyplot(dict_figuras_laudo[f"Ensemble Ângulos ({art_selecionada})"])
        with col_g2:
            st.pyplot(dict_figuras_laudo[f"Função de Simetria ({art_selecionada})"])

# --- 3. ATUAÇÃO MUSCULAR & POTÊNCIA (ENSEMBLE AVERAGE & SIMETRIA) ---
st.header("3. Atuação e Potência Muscular: Ensemble Average & Função de Simetria")

grupos_padrao = {
    'Quadríceps': ['vasint', 'vaslat', 'vasmed', 'recfem', 'rect_fem', 'vastus', 'quad'],
    'Glúteo Médio': ['glmed1', 'glmed2', 'glmed3', 'glut_med'],
    'Glúteo Máximo': ['glmax1', 'glmax2', 'glmax3', 'glut_max'],
    'Isquiotibiais': ['bflh', 'bfsh', 'semimem', 'semiten', 'biceps_fem', 'hamstring'],
    'Gastrocnêmio': ['gaslat', 'gasmed'],
    'Solear': ['soleus', 'sol'],
    'Tensor da Fáscia Lata': ['tfl', 'tensor']
}

# 1. PROCESSAMENTO DAS FORÇAS MUSCULARES
if file_muscles is not None and len(hs_r) >= 2 and len(hs_l) >= 2:
    df_musc, _ = carregar_arquivo_opensim(file_muscles)
    colunas_musc = [c for c in df_musc.columns if c != 'time']
    músculos_disponiveis = sorted(list(set([c[:-2] for c in colunas_musc if c.endswith('_r') or c.endswith('_l')])))
    
    for m_sel in músculos_disponiveis:
        m_r, m_l = f"{m_sel}_r", f"{m_sel}_l"
        if m_r in df_musc.columns and m_l in df_musc.columns:
            ens_mr = extrair_ensemble_average(df_musc, 'time', {m_sel: m_r}, hs_r, peso_corporal)
            ens_ml = extrair_ensemble_average(df_musc, 'time', {m_sel: m_l}, hs_l, peso_corporal)
            
            mr_val, mr_std = ens_mr[f'{m_sel}_mean'], ens_mr[f'{m_sel}_std']
            ml_val, ml_std = ens_ml[f'{m_sel}_mean'], ens_ml[f'{m_sel}_std']
            perc = ens_mr['% Ciclo']
            sp_m = symmetry_plot(mr_val, ml_val)
            
            fig_m, ax_m = plt.subplots(figsize=(6, 3.5))
            ax_m.plot(perc, mr_val, label='Direito (Média)', color='#02579B', linewidth=2)
            ax_m.fill_between(perc, mr_val - mr_std, mr_val + mr_std, color='#02579B', alpha=0.2)
            
            ax_m.plot(perc, ml_val, label='Esquerdo (Média)', color='#ff9933', linewidth=2)
            ax_m.fill_between(perc, ml_val - ml_std, ml_val + ml_std, color='#ff9933', alpha=0.2)
            
            ax_m.set_xlabel("% do Ciclo de Marcha")
            ax_m.set_ylabel("Ativação / Força (BW)")
            ax_m.legend()
            ax_m.grid(True, linestyle='--', alpha=0.5)
            plt.close(fig_m)
            dict_figuras_laudo[f"Ensemble Muscular ({m_sel})"] = fig_m
            
            fig_msym, ax_msym = plt.subplots(figsize=(6, 3.5))
            ax_msym.scatter(sp_m['time'], sp_m['score'], color='gray', alpha=0.3, s=15, label='Bruto')
            ax_msym.plot(sp_m['time'], sp_m['score_smoothed'], color='red', linewidth=2.5, label='Ajustado')
            ax_msym.axvline(x=40, color='gray', linestyle='--', label='Off-foot (40%)')
            ax_msym.set_xlabel("Gait Cycle (%)")
            ax_msym.set_ylabel("Symmetry Function (%)")
            ax_msym.legend()
            ax_msym.grid(True, linestyle='--', alpha=0.5)
            plt.close(fig_msym)
            dict_figuras_laudo[f"Simetria Muscular ({m_sel})"] = fig_msym

    for nome_grp, palavras_chave in grupos_padrao.items():
        muscs_grp = [m for m in músculos_disponiveis if any(k in m.lower() for k in palavras_chave)]
        if muscs_grp:
            dict_r = {nome_grp: [f"{m}_r" for m in muscs_grp]}
            dict_l = {nome_grp: [f"{m}_l" for m in muscs_grp]}
            
            ens_gr = extrair_ensemble_average(df_musc, 'time', dict_r, hs_r, peso_corporal)
            ens_gl = extrair_ensemble_average(df_musc, 'time', dict_l, hs_l, peso_corporal)
            
            if ens_gr and ens_gl and f'{nome_grp}_mean' in ens_gr:
                gr_val, gr_std = ens_gr[f'{nome_grp}_mean'], ens_gr[f'{nome_grp}_std']
                gl_val, gl_std = ens_gl[f'{nome_grp}_mean'], ens_gl[f'{nome_grp}_std']
                perc = ens_gr['% Ciclo']
                sp_g = symmetry_plot(gr_val, gl_val)
                
                fig_g, ax_g = plt.subplots(figsize=(6, 3.5))
                ax_g.plot(perc, gr_val, label='Direito (Média)', color='#02579B', linewidth=2)
                ax_g.fill_between(perc, gr_val - gr_std, gr_val + gr_std, color='#02579B', alpha=0.2)
                
                ax_g.plot(perc, gl_val, label='Esquerdo (Média)', color='#ff9933', linewidth=2)
                ax_g.fill_between(perc, gl_val - gl_std, gl_val + gl_std, color='#ff9933', alpha=0.2)
                
                ax_g.set_xlabel("% do Ciclo de Marcha")
                ax_g.set_ylabel("Ativação / Força (BW)")
                ax_g.legend()
                ax_g.grid(True, linestyle='--', alpha=0.5)
                plt.close(fig_g)
                dict_figuras_laudo[f"Ensemble Muscular ({nome_grp})"] = fig_g
                
                fig_gsym, ax_gsym = plt.subplots(figsize=(6, 3.5))
                ax_gsym.scatter(sp_g['time'], sp_g['score'], color='gray', alpha=0.3, s=15, label='Bruto')
                ax_gsym.plot(sp_g['time'], sp_gsym_smoothed := sp_g['score_smoothed'], color='red', linewidth=2.5, label='Ajustado')
                ax_gsym.axvline(x=40, color='gray', linestyle='--', label='Off-foot (40%)')
                ax_gsym.set_xlabel("Gait Cycle (%)")
                ax_gsym.set_ylabel("Symmetry Function (%)")
                ax_gsym.legend()
                ax_gsym.grid(True, linestyle='--', alpha=0.5)
                plt.close(fig_gsym)
                dict_figuras_laudo[f"Simetria Muscular ({nome_grp})"] = fig_gsym

# 2. PROCESSAMENTO DA POTÊNCIA MUSCULAR (POWER)
if file_power is not None and len(hs_r) >= 2 and len(hs_l) >= 2:
    df_pow, _ = carregar_arquivo_opensim(file_power)
    colunas_pow = [c for c in df_pow.columns if c != 'time']
    musculos_pow = sorted(list(set([c[:-2] for c in colunas_pow if c.endswith('_r') or c.endswith('_l')])))
    
    for m_sel in musculos_pow:
        m_r, m_l = f"{m_sel}_r", f"{m_sel}_l"
        if m_r in df_pow.columns and m_l in df_pow.columns:
            ens_pr = extrair_ensemble_average(df_pow, 'time', {m_sel: m_r}, hs_r, peso_corporal)
            ens_pl = extrair_ensemble_average(df_pow, 'time', {m_sel: m_l}, hs_l, peso_corporal)
            
            pr_val, pr_std = ens_pr[f'{m_sel}_mean'], ens_pr[f'{m_sel}_std']
            pl_val, pl_std = ens_pl[f'{m_sel}_mean'], ens_pl[f'{m_sel}_std']
            perc = ens_pr['% Ciclo']
            
            fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
            ax_p.plot(perc, pr_val, label='Direito (Média)', color='#02579B', linewidth=2)
            ax_p.fill_between(perc, pr_val - pr_std, pr_val + pr_std, color='#02579B', alpha=0.2)
            
            ax_p.plot(perc, pl_val, label='Esquerdo (Média)', color='#ff9933', linewidth=2)
            ax_p.fill_between(perc, pl_val - pl_std, pl_val + pl_std, color='#ff9933', alpha=0.2)
            
            ax_p.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            ax_p.set_xlabel("% do Ciclo de Marcha")
            ax_p.set_ylabel("Potência Muscular (W/kg)")
            ax_p.legend()
            ax_p.grid(True, linestyle='--', alpha=0.5)
            plt.close(fig_p)
            dict_figuras_laudo[f"Potência Muscular ({m_sel})"] = fig_p

    for nome_grp, palavras_chave in grupos_padrao.items():
        muscs_grp = [m for m in musculos_pow if any(k in m.lower() for k in palavras_chave)]
        if muscs_grp:
            dict_r = {nome_grp: [f"{m}_r" for m in muscs_grp]}
            dict_l = {nome_grp: [f"{m}_l" for m in muscs_grp]}
            
            ens_pr = extrair_ensemble_average(df_pow, 'time', dict_r, hs_r, peso_corporal)
            ens_pl = extrair_ensemble_average(df_pow, 'time', dict_l, hs_l, peso_corporal)
            
            if ens_pr and ens_pl and f'{nome_grp}_mean' in ens_pr:
                pr_val, pr_std = ens_pr[f'{nome_grp}_mean'], ens_pr[f'{nome_grp}_std']
                pl_val, pl_std = ens_pl[f'{nome_grp}_mean'], ens_pl[f'{nome_grp}_std']
                perc = ens_pr['% Ciclo']
                
                fig_p, ax_p = plt.subplots(figsize=(6, 3.5))
                ax_p.plot(perc, pr_val, label='Direito (Média)', color='#02579B', linewidth=2)
                ax_p.fill_between(perc, pr_val - pr_std, pr_val + pr_std, color='#02579B', alpha=0.2)
                
                ax_p.plot(perc, pl_val, label='Esquerdo (Média)', color='#ff9933', linewidth=2)
                ax_p.fill_between(perc, pl_val - pl_std, pl_val + pl_std, color='#ff9933', alpha=0.2)
                
                ax_p.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_p.set_xlabel("% do Ciclo de Marcha")
                ax_p.set_ylabel("Potência Muscular (W/kg)")
                ax_p.legend()
                ax_p.grid(True, linestyle='--', alpha=0.5)
                plt.close(fig_p)
                dict_figuras_laudo[f"Potência Muscular ({nome_grp})"] = fig_p

# --- EXIBIÇÃO EM TELA COM 3 COLUNAS (FORÇA, SIMETRIA E POTÊNCIA) ---
opcoes_musculares_tela = [k.replace("Ensemble Muscular (", "").replace(")", "") for k in dict_figuras_laudo.keys() if k.startswith("Ensemble Muscular (")]

if opcoes_musculares_tela:
    musculo_ou_grupo_sel = st.selectbox("Selecione o músculo ou grupo muscular para visualizar na tela:", opcoes_musculares_tela)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.caption("**1. Força / Ativação (BW)**")
        if f"Ensemble Muscular ({musculo_ou_grupo_sel})" in dict_figuras_laudo:
            st.pyplot(dict_figuras_laudo[f"Ensemble Muscular ({musculo_ou_grupo_sel})"])
            
    with col_m2:
        st.caption("**2. Função de Simetria (%)**")
        if f"Simetria Muscular ({musculo_ou_grupo_sel})" in dict_figuras_laudo:
            st.pyplot(dict_figuras_laudo[f"Simetria Muscular ({musculo_ou_grupo_sel})"])
            
    with col_m3:
        st.caption("**3. Potência Muscular (W/kg)**")
        if f"Potência Muscular ({musculo_ou_grupo_sel})" in dict_figuras_laudo:
            st.pyplot(dict_figuras_laudo[f"Potência Muscular ({musculo_ou_grupo_sel})"])
        elif file_power is None:
            st.info("Suba o arquivo `4. Potência Muscular` na barra lateral para liberar este gráfico.")

# --- 4. SEÇÃO DE GERAÇÃO DO LAUDO ---
st.header("4. Gerador de Laudo Biomecânico (PDF)")

col_l1, col_l2 = st.columns([1, 1])

with col_l1:
    nome_atleta = st.text_input("Nome do Avaliado / Atleta:", "Atleta Exemplo")
    obs_clinicas = st.text_area("Observações / Diagnóstico Biomecânico:", "Atleta apresenta simetria preservada com variabilidade padrão em torno da Média Ensemble.")

with col_l2:
    st.subheader("Seleção de Gráficos para o Laudo")
    graficos_disponiveis = list(dict_figuras_laudo.keys())
    
    if graficos_disponiveis:
        termos_default_laudo = [
            'hip_flexion', 'knee_angle', 'ankle_angle',
            'Quadríceps', 'recfem', 'vaslat',
            'Solear', 'soleus',
            'Glúteo Máximo', 'glmax',
            'Glúteo Médio', 'glmed',
            'Isquiotibiais', 'bflh',
            'Tensor da Fáscia Lata', 'tfl'
        ]
        
        sugestao_laudo = []
        for g in graficos_disponiveis:
            if any(t.lower() in g.lower() for t in termos_default_laudo):
                sugestao_laudo.append(g)
                
        graficos_escolhidos = st.multiselect(
            "Escolha quais gráficos serão incluídos no PDF:",
            graficos_disponiveis,
            default=sugestao_laudo if sugestao_laudo else graficos_disponiveis
        )
    else:
        st.info("Carregue os arquivos para disponibilizar os gráficos.")
        graficos_escolhidos = []

st.markdown("---")

if st.button("📄 Gerar Laudo PDF Completo", type="primary"):
    if not graficos_escolhidos:
        st.error("Selecione ao menos um gráfico para exportar.")
    else:
        figs_para_pdf = [(title, dict_figuras_laudo[title]) for title in graficos_escolhidos]
        pdf_bytes = gerar_pdf_laudo(nome_atleta, vel_esteira_kmh, obs_clinicas, dados_et, figs_para_pdf)
        
        st.success("Laudo em PDF gerado com sucesso!")
        st.download_button(
            label="⬇️ Baixar Laudo Biomecânico (PDF)",
            data=pdf_bytes,
            file_name=f"laudo_biomecanico_{nome_atleta.lower().replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
