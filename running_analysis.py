"""
Running Biomechanics & Power - OpenSim
App Streamlit com suporte a MÚLTIPLAS AVALIAÇÕES e MODO COMPARATIVO.

Recursos:
- Ensemble average estilo BMClab (média±SD ddof=1 ou mediana±IQR)
- Interpolação linear ou spline cúbica
- Suavização Savitzky-Golay opcional
- Cinemática articular (IK) + Pelvis (tilt/list/rotation)
- Velocidades angulares (por articulação e do pelvis)
- Força/ativação e potência muscular (por músculo e por grupo)
- Gerador de laudo em PDF
"""

import io
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy.interpolate import CubicSpline

import streamlit as st

# ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ============================================================
# 0. CONFIGURAÇÃO GLOBAL DO ENSEMBLE
# ============================================================
N_PONTOS_CICLO = 101
INTERP_METHOD = 'linear'
SUAVIZAR_CICLOS = False
SAVGOL_WINDOW = 11
SAVGOL_ORDER = 2
METRICA_ENSEMBLE = 'mean_sd'


# ============================================================
# 1. PARSER E PROCESSAMENTO DE SINAL
# ============================================================
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
    if not (0 < normal_cutoff < 1):
        return sinal
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, sinal)


# ============================================================
# 2. MÉTODOS DE LEITCH ET AL. (2011)
# ============================================================
def extrair_heel_strikes_leitch(df_pos, sufixo_lado='r'):
    col_y = f'calcn_{sufixo_lado}_Y'
    if col_y not in df_pos.columns:
        return np.array([])

    fs, _ = extrair_freq_amostragem(df_pos)
    calcn_y = filtrar_sinal(df_pos[col_y].values, fs=fs, fc=8.0)
    v_y = np.gradient(calcn_y) * fs

    min_y = np.min(calcn_y)
    max_y = np.max(calcn_y)
    threshold_y = min_y + 0.35 * (max_y - min_y)
    min_dist_frames = max(1, int(0.25 * fs))

    peaks, _ = find_peaks(
        -calcn_y, distance=min_dist_frames,
        height=-threshold_y, prominence=0.015
    )

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
    duracoes = [
        tempo[hs_indices[i + 1]] - tempo[hs_indices[i]]
        for i in range(len(hs_indices) - 1)
    ]
    t_passada_medio = float(np.mean(duracoes))

    vel_ms = vel_esteira_kmh / 3.6
    comp_passada_medio = vel_ms * t_passada_medio
    cadencia_media = (60.0 / t_passada_medio) * 2.0 if t_passada_medio > 0 else 0.0

    return t_passada_medio, comp_passada_medio, vel_ms, cadencia_media


# ============================================================
# 3. ENSEMBLE AVERAGE (estilo BMClab)
# ============================================================
def _interpolar_ciclo(t_ciclo, sinal_ciclo, t_norm, metodo='linear'):
    if metodo == 'cubic' and len(t_ciclo) >= 4:
        try:
            cs = CubicSpline(t_ciclo, sinal_ciclo)
            return cs(t_norm)
        except Exception:
            return np.interp(t_norm, t_ciclo, sinal_ciclo)
    return np.interp(t_norm, t_ciclo, sinal_ciclo)


def _suavizar(sinal, window=11, order=2):
    n = len(sinal)
    if n < 5:
        return sinal
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return sinal
    if order >= w:
        order = w - 1
    try:
        return savgol_filter(sinal, window_length=w, polyorder=order)
    except Exception:
        return sinal


def extrair_ensemble_average(df, coluna_tempo, dict_colunas_ou_grupos,
                             hs_indices, peso_corporal=1.0,
                             metodo_interp: str = None,
                             suavizar: bool = None,
                             metrica: str = None):
    """
    Ensemble average no estilo BMClab.
    Retorna dict com '% Ciclo', '<var>_mean', '<var>_std',
    '<var>_median', '<var>_q1', '<var>_q3', '<var>_cycles'.
    """
    if len(hs_indices) < 2:
        return None

    metodo_interp = metodo_interp or INTERP_METHOD
    suavizar = SUAVIZAR_CICLOS if suavizar is None else suavizar
    metrica = metrica or METRICA_ENSEMBLE

    percentual_ciclo = np.linspace(0, 100, N_PONTOS_CICLO)
    tempo = df[coluna_tempo].values
    resultados = {'% Ciclo': percentual_ciclo}

    for nome_var, colunas in dict_colunas_ou_grupos.items():
        if isinstance(colunas, str):
            colunas = [colunas]

        cols_existentes = [c for c in colunas if c in df.columns]
        if not cols_existentes:
            continue

        soma_bruta = df[cols_existentes].sum(axis=1).values / peso_corporal
        ciclos_matriz = []

        for i in range(len(hs_indices) - 1):
            idx_i = int(hs_indices[i])
            idx_f = int(hs_indices[i + 1])

            if idx_f <= idx_i:
                continue

            t_ciclo = tempo[idx_i:idx_f]
            if len(t_ciclo) < 3:
                continue

            t_norm = np.linspace(t_ciclo[0], t_ciclo[-1], N_PONTOS_CICLO)
            sinal_ciclo = soma_bruta[idx_i:idx_f]

            sinal_interp = _interpolar_ciclo(t_ciclo, sinal_ciclo,
                                             t_norm, metodo=metodo_interp)

            if suavizar:
                sinal_interp = _suavizar(sinal_interp,
                                         SAVGOL_WINDOW, SAVGOL_ORDER)

            ciclos_matriz.append(sinal_interp)

        if not ciclos_matriz:
            continue

        ciclos_matriz = np.array(ciclos_matriz)

        resultados[f'{nome_var}_mean'] = np.mean(ciclos_matriz, axis=0)
        if ciclos_matriz.shape[0] > 1:
            resultados[f'{nome_var}_std'] = np.std(ciclos_matriz,
                                                   axis=0, ddof=1)
        else:
            resultados[f'{nome_var}_std'] = np.zeros(ciclos_matriz.shape[1])

        resultados[f'{nome_var}_median'] = np.median(ciclos_matriz, axis=0)
        resultados[f'{nome_var}_q1'] = np.percentile(ciclos_matriz, 25, axis=0)
        resultados[f'{nome_var}_q3'] = np.percentile(ciclos_matriz, 75, axis=0)
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

    window = min(15, len(diff_pct) if len(diff_pct) % 2 == 1 else len(diff_pct) - 1)
    if window < 5:
        smoothed = diff_pct
    else:
        smoothed = savgol_filter(diff_pct, window_length=window, polyorder=2)

    return pd.DataFrame({
        'time': time_idx,
        'score': diff_pct,
        'score_smoothed': smoothed
    })


# ============================================================
# 4. DATACLASS DE AVALIAÇÃO
# ============================================================
@dataclass
class Avaliacao:
    id: str
    nome: str
    velocidade: float
    peso: float
    data: str

    df_pos: Optional[pd.DataFrame] = None
    df_ik: Optional[pd.DataFrame] = None
    df_angvel: Optional[pd.DataFrame] = None
    df_musc: Optional[pd.DataFrame] = None
    df_pow: Optional[pd.DataFrame] = None

    in_degrees: bool = True
    fs: float = 100.0

    hs_r: np.ndarray = field(default_factory=lambda: np.array([]))
    hs_l: np.ndarray = field(default_factory=lambda: np.array([]))

    dados_et: dict = field(default_factory=dict)

    ensembles: Dict[str, dict] = field(default_factory=dict)

    @property
    def n_ciclos_r(self):
        return max(0, len(self.hs_r) - 1)

    @property
    def n_ciclos_l(self):
        return max(0, len(self.hs_l) - 1)


def gerar_id(nome, vel, data):
    raw = f"{nome}|{vel}|{data}".encode()
    return hashlib.md5(raw).hexdigest()[:8]


# ============================================================
# 5. GRUPOS MUSCULARES PADRÃO
# ============================================================
GRUPOS_PADRAO = {
    'Quadríceps': ['vasint', 'vaslat', 'vasmed', 'recfem', 'rect_fem', 'vastus', 'quad'],
    'Glúteo Médio': ['glmed1', 'glmed2', 'glmed3', 'glut_med'],
    'Glúteo Máximo': ['glmax1', 'glmax2', 'glmax3', 'glut_max'],
    'Isquiotibiais': ['bflh', 'bfsh', 'semimem', 'semiten', 'biceps_fem', 'hamstring'],
    'Gastrocnêmio': ['gaslat', 'gasmed'],
    'Solear': ['soleus', 'sol'],
    'Tensor da Fáscia Lata': ['tfl', 'tensor'],
}


# ============================================================
# 6. PROCESSAMENTO DE UMA AVALIAÇÃO
# ============================================================
def _adicionar_ensemble(av: Avaliacao, chave: str, label: str, tipo: str,
                        ens_r: dict, ens_l: dict,
                        nome_var_r: str, nome_var_l: str):
    if not ens_r or not ens_l:
        return
    if f'{nome_var_r}_mean' not in ens_r or f'{nome_var_l}_mean' not in ens_l:
        return

    av.ensembles[chave] = {
        'label': label,
        'tipo': tipo,
        'perc': ens_r['% Ciclo'],
        'mean_r': ens_r[f'{nome_var_r}_mean'],
        'std_r': ens_r[f'{nome_var_r}_std'],
        'mean_l': ens_l[f'{nome_var_l}_mean'],
        'std_l': ens_l[f'{nome_var_l}_std'],
        'median_r': ens_r.get(f'{nome_var_r}_median'),
        'q1_r': ens_r.get(f'{nome_var_r}_q1'),
        'q3_r': ens_r.get(f'{nome_var_r}_q3'),
        'median_l': ens_l.get(f'{nome_var_l}_median'),
        'q1_l': ens_l.get(f'{nome_var_l}_q1'),
        'q3_l': ens_l.get(f'{nome_var_l}_q3'),
        'cycles_r': ens_r.get(f'{nome_var_r}_cycles'),
        'cycles_l': ens_l.get(f'{nome_var_l}_cycles'),
    }


def _detectar_pelvis(colunas):
    """
    Dado um conjunto de colunas, retorna dict:
        {'pelvis_tilt': <coluna>, 'pelvis_list': <coluna>, 'pelvis_rotation': <coluna>}
    Escolhe uma coluna por movimento (prefere a sem sufixo _r/_l).
    """
    movimentos = []
    for col in colunas:
        low = col.lower()
        if 'pelv' not in low:
            continue
        if 'tilt' in low:
            movimentos.append(('pelvis_tilt', col))
        elif 'list' in low:
            movimentos.append(('pelvis_list', col))
        elif 'rotation' in low or '_rot' in low or low.endswith('rot'):
            movimentos.append(('pelvis_rotation', col))

    vistos = {}
    for nome_mov, col in movimentos:
        if nome_mov not in vistos:
            vistos[nome_mov] = col
        else:
            atual = vistos[nome_mov]
            if col.endswith(('_r', '_l')) and not atual.endswith(('_r', '_l')):
                continue
            if not col.endswith(('_r', '_l')) and atual.endswith(('_r', '_l')):
                vistos[nome_mov] = col

    return vistos


def _processar_pelvis_generico(av: Avaliacao, df: pd.DataFrame,
                               prefixo_chave: str,
                               prefixo_label: str,
                               tipo_ensemble: str):
    """Processa tilt/list/rotation do pelvis para um dado DataFrame."""
    colunas = [c for c in df.columns if c != 'time']
    vistos = _detectar_pelvis(colunas)

    for nome_mov, col_pelvis in vistos.items():
        if col_pelvis not in df.columns:
            continue
        ens_r = extrair_ensemble_average(df, 'time', {'v': col_pelvis}, av.hs_r)
        ens_l = extrair_ensemble_average(df, 'time', {'v': col_pelvis}, av.hs_l)
        label = nome_mov.replace('pelvis_', '').replace('_', ' ').title()
        _adicionar_ensemble(
            av,
            f"{prefixo_chave}pelvis::{nome_mov}",
            f"{prefixo_label}Pelvis {label}",
            tipo_ensemble,
            ens_r, ens_l, 'v', 'v'
        )

    return set(vistos.values())


def _processar_pares_generico(av: Avaliacao, df: pd.DataFrame,
                              colunas_excluidas: set,
                              prefixo_chave: str,
                              prefixo_label: str,
                              tipo_ensemble: str):
    """Processa pares R/L de um DataFrame qualquer."""
    colunas = [c for c in df.columns if c != 'time']

    pares = sorted(list(set([
        c[:-2] for c in colunas
        if (c.endswith('_r') or c.endswith('_l'))
        and c not in colunas_excluidas
    ])))

    for art in pares:
        col_r, col_l = f"{art}_r", f"{art}_l"
        if col_r not in df.columns or col_l not in df.columns:
            continue
        ens_r = extrair_ensemble_average(df, 'time', {'v': col_r}, av.hs_r)
        ens_l = extrair_ensemble_average(df, 'time', {'v': col_l}, av.hs_l)
        label = art.replace('_', ' ').title()
        _adicionar_ensemble(av, f"{prefixo_chave}{art}",
                            f"{prefixo_label}{label}",
                            tipo_ensemble,
                            ens_r, ens_l, 'v', 'v')


def _processar_ik(av: Avaliacao):
    if av.df_ik is None or av.n_ciclos_r < 1 or av.n_ciclos_l < 1:
        return

    colunas = [c for c in av.df_ik.columns if c != 'time']
    if not av.in_degrees and colunas:
        av.df_ik[colunas] = av.df_ik[colunas] * (180.0 / np.pi)

    excluidas = _processar_pelvis_generico(
        av, av.df_ik,
        prefixo_chave="pelvis::",
        prefixo_label="",
        tipo_ensemble='pelvis'
    )
    _processar_pares_generico(
        av, av.df_ik,
        colunas_excluidas=excluidas,
        prefixo_chave="ik::",
        prefixo_label="",
        tipo_ensemble='ik'
    )


def _processar_vel_angular(av: Avaliacao):
    if av.df_angvel is None or av.n_ciclos_r < 1 or av.n_ciclos_l < 1:
        return

    colunas = [c for c in av.df_angvel.columns if c != 'time']
    if not av.in_degrees and colunas:
        av.df_angvel[colunas] = av.df_angvel[colunas] * (180.0 / np.pi)

    excluidas = _processar_pelvis_generico(
        av, av.df_angvel,
        prefixo_chave="angvel::pelvis::",
        prefixo_label="Vel. Angular — ",
        tipo_ensemble='pelvis_angvel'
    )
    _processar_pares_generico(
        av, av.df_angvel,
        colunas_excluidas=excluidas,
        prefixo_chave="angvel::",
        prefixo_label="Vel. Angular — ",
        tipo_ensemble='angvel'
    )


def _processar_musculos(av: Avaliacao):
    if av.df_musc is None or av.n_ciclos_r < 1 or av.n_ciclos_l < 1:
        return

    colunas = [c for c in av.df_musc.columns if c != 'time']
    musculos = sorted(list(set([
        c[:-2] for c in colunas if c.endswith('_r') or c.endswith('_l')
    ])))

    for m in musculos:
        m_r, m_l = f"{m}_r", f"{m}_l"
        if m_r not in av.df_musc.columns or m_l not in av.df_musc.columns:
            continue
        ens_r = extrair_ensemble_average(av.df_musc, 'time', {'v': m_r},
                                         av.hs_r, av.peso)
        ens_l = extrair_ensemble_average(av.df_musc, 'time', {'v': m_l},
                                         av.hs_l, av.peso)
        _adicionar_ensemble(av, f"musc::{m}", m, 'musc', ens_r, ens_l, 'v', 'v')

    for nome_grp, chaves in GRUPOS_PADRAO.items():
        ms = [m for m in musculos if any(k in m.lower() for k in chaves)]
        if not ms:
            continue
        cols_r = [f"{m}_r" for m in ms if f"{m}_r" in av.df_musc.columns]
        cols_l = [f"{m}_l" for m in ms if f"{m}_l" in av.df_musc.columns]
        if not cols_r or not cols_l:
            continue
        ens_r = extrair_ensemble_average(av.df_musc, 'time', {'v': cols_r},
                                         av.hs_r, av.peso)
        ens_l = extrair_ensemble_average(av.df_musc, 'time', {'v': cols_l},
                                         av.hs_l, av.peso)
        _adicionar_ensemble(av, f"muscgrp::{nome_grp}", nome_grp, 'musc',
                            ens_r, ens_l, 'v', 'v')


def _processar_potencia(av: Avaliacao):
    if av.df_pow is None or av.n_ciclos_r < 1 or av.n_ciclos_l < 1:
        return

    colunas = [c for c in av.df_pow.columns if c != 'time']
    musculos = sorted(list(set([
        c[:-2] for c in colunas if c.endswith('_r') or c.endswith('_l')
    ])))

    for m in musculos:
        m_r, m_l = f"{m}_r", f"{m}_l"
        if m_r not in av.df_pow.columns or m_l not in av.df_pow.columns:
            continue
        ens_r = extrair_ensemble_average(av.df_pow, 'time', {'v': m_r},
                                         av.hs_r, av.peso)
        ens_l = extrair_ensemble_average(av.df_pow, 'time', {'v': m_l},
                                         av.hs_l, av.peso)
        _adicionar_ensemble(av, f"pow::{m}", m, 'pow', ens_r, ens_l, 'v', 'v')

    for nome_grp, chaves in GRUPOS_PADRAO.items():
        ms = [m for m in musculos if any(k in m.lower() for k in chaves)]
        if not ms:
            continue
        cols_r = [f"{m}_r" for m in ms if f"{m}_r" in av.df_pow.columns]
        cols_l = [f"{m}_l" for m in ms if f"{m}_l" in av.df_pow.columns]
        if not cols_r or not cols_l:
            continue
        ens_r = extrair_ensemble_average(av.df_pow, 'time', {'v': cols_r},
                                         av.hs_r, av.peso)
        ens_l = extrair_ensemble_average(av.df_pow, 'time', {'v': cols_l},
                                         av.hs_l, av.peso)
        _adicionar_ensemble(av, f"powgrp::{nome_grp}", nome_grp, 'pow',
                            ens_r, ens_l, 'v', 'v')


def processar_avaliacao(nome, vel, peso, data,
                        f_pos, f_ik, f_angvel, f_musc, f_pow) -> Avaliacao:
    av = Avaliacao(
        id=gerar_id(nome, vel, data),
        nome=nome, velocidade=vel, peso=peso, data=data
    )

    if f_pos is not None:
        av.df_pos, _ = carregar_arquivo_opensim(f_pos)
        av.fs, _ = extrair_freq_amostragem(av.df_pos)
        av.hs_r = extrair_heel_strikes_leitch(av.df_pos, 'r')
        av.hs_l = extrair_heel_strikes_leitch(av.df_pos, 'l')

        if av.n_ciclos_r >= 1 and av.n_ciclos_l >= 1:
            t_r, c_r, v_r, cad_r = calcular_espaco_temporais_membro(
                av.df_pos, av.hs_r, vel)
            t_l, c_l, v_l, cad_l = calcular_espaco_temporais_membro(
                av.df_pos, av.hs_l, vel)
            av.dados_et = {
                't_r': t_r, 'comp_r': c_r, 'v_r': v_r, 'cad_r': cad_r,
                't_l': t_l, 'comp_l': c_l, 'v_l': v_l, 'cad_l': cad_l,
            }

    if f_ik is not None and av.n_ciclos_r >= 1 and av.n_ciclos_l >= 1:
        av.df_ik, av.in_degrees = carregar_arquivo_opensim(f_ik)
        _processar_ik(av)

    if f_angvel is not None and av.n_ciclos_r >= 1 and av.n_ciclos_l >= 1:
        av.df_angvel, av.in_degrees = carregar_arquivo_opensim(f_angvel)
        _processar_vel_angular(av)

    if f_musc is not None and av.n_ciclos_r >= 1 and av.n_ciclos_l >= 1:
        av.df_musc, _ = carregar_arquivo_opensim(f_musc)
        _processar_musculos(av)

    if f_pow is not None and av.n_ciclos_r >= 1 and av.n_ciclos_l >= 1:
        av.df_pow, _ = carregar_arquivo_opensim(f_pow)
        _processar_potencia(av)

    return av


# ============================================================
# 7. PLOT HELPERS
# ============================================================
def _plot_ensemble_generico(ax, perc, mean, std, median, q1, q3,
                            color, label, metrica='mean_sd'):
    if metrica == 'median_iqr' and median is not None:
        lo = median - q1
        hi = q3 - median
        ax.fill_between(perc, median - lo, median + hi,
                        color=color, alpha=0.2)
        ax.plot(perc, median, color=color, linewidth=2, label=label)
    else:
        ax.fill_between(perc, mean - std, mean + std,
                        color=color, alpha=0.2)
        ax.plot(perc, mean, color=color, linewidth=2, label=label)


def _rotulos_lado(av: Avaliacao, chave: str):
    tipo = av.ensembles[chave].get('tipo', '')
    if tipo in ('pelvis', 'pelvis_angvel'):
        return 'HS Direito', 'HS Esquerdo'
    return 'Direito', 'Esquerdo'


def plot_ensemble_individual(av: Avaliacao, chave: str,
                             metrica: str = 'mean_sd') -> plt.Figure:
    e = av.ensembles[chave]
    rot_r, rot_l = _rotulos_lado(av, chave)

    fig, ax = plt.subplots(figsize=(6, 3.5))

    _plot_ensemble_generico(
        ax, e['perc'], e['mean_r'], e['std_r'],
        e.get('median_r'), e.get('q1_r'), e.get('q3_r'),
        color='#02579B', label=rot_r, metrica=metrica
    )
    _plot_ensemble_generico(
        ax, e['perc'], e['mean_l'], e['std_l'],
        e.get('median_l'), e.get('q1_l'), e.get('q3_l'),
        color='#ff9933', label=rot_l, metrica=metrica
    )

    ax.set_xlabel("% do Ciclo de Marcha")
    ax.set_ylabel("Valor")
    ax.set_title(e['label'])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def plot_simetria(av: Avaliacao, chave: str,
                  metrica: str = 'mean_sd') -> plt.Figure:
    e = av.ensembles[chave]
    if metrica == 'median_iqr' and e.get('median_r') is not None:
        curva_r = e['median_r']
        curva_l = e['median_l']
    else:
        curva_r = e['mean_r']
        curva_l = e['mean_l']

    sp = symmetry_plot(curva_r, curva_l)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(sp['time'], sp['score'], color='gray', alpha=0.3, s=15,
               label='Bruto')
    ax.plot(sp['time'], sp['score_smoothed'], color='blue', linewidth=2.5,
            label='Ajustado')
    ax.axvline(x=40, color='gray', linestyle='--', label='Off-foot (40%)')
    ax.set_xlabel("Gait Cycle (%)")
    ax.set_ylabel("Symmetry Function (%)")
    ax.set_title(f"Simetria — {e['label']}")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


def plot_comparativo(avaliacoes: List[Avaliacao], chave: str,
                     lado: str = 'both',
                     metrica: str = 'mean_sd') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cores = plt.cm.tab10(np.linspace(0, 1, max(len(avaliacoes), 2)))

    for i, av in enumerate(avaliacoes):
        if chave not in av.ensembles:
            continue
        e = av.ensembles[chave]
        cor = cores[i % len(cores)]
        rotulo_base = f"{av.nome} ({av.velocidade:.1f} km/h)"
        rot_r, rot_l = _rotulos_lado(av, chave)

        for lado_atual, sufixo_rotulo, ls in (
            ('r', rot_r, '-'),
            ('l', rot_l, '--'),
        ):
            if lado not in ('both', lado_atual):
                continue

            if metrica == 'median_iqr' and e.get(f'median_{lado_atual}') is not None:
                central = e[f'median_{lado_atual}']
                q1 = e[f'q1_{lado_atual}']
                q3 = e[f'q3_{lado_atual}']
                ax.fill_between(e['perc'], central - (central - q1),
                                central + (q3 - central),
                                color=cor, alpha=0.12)
            else:
                central = e[f'mean_{lado_atual}']
                std = e[f'std_{lado_atual}']
                ax.fill_between(e['perc'], central - std, central + std,
                                color=cor, alpha=0.12)

            rotulo = f"{rotulo_base} - {sufixo_rotulo}" if lado == 'both' else rotulo_base
            ax.plot(e['perc'], central, color=cor, linewidth=2,
                    linestyle=ls, label=rotulo)

    ax.set_xlabel("% do Ciclo de Marcha")
    ax.set_ylabel("Valor")
    ax.set_title(f"Comparação — {chave}")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


# ============================================================
# 8. GERADOR DE PDF
# ============================================================
def _fig_para_buffer(fig: plt.Figure) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf


def gerar_pdf_laudo(nome_atleta, observacoes, dados_et,
                    figuras_selecionadas: List[Tuple[str, plt.Figure]]):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=36, leftMargin=36,
                            topMargin=36, bottomMargin=36)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                 fontSize=16, leading=20, alignment=1,
                                 textColor=colors.HexColor('#1E3A8A'))
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'],
                                   fontSize=11, leading=15,
                                   textColor=colors.HexColor('#1E3A8A'))
    normal_style = styles['Normal']

    story.append(Paragraph("LAUDO DE ANÁLISE BIOMECÂNICA DA CORRIDA",
                           title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Atleta / Avaliado:</b> {nome_atleta}",
                           normal_style))
    story.append(Paragraph(
        f"<b>Data da Análise:</b> {pd.Timestamp.now().strftime('%d/%m/%Y')}",
        normal_style))
    story.append(Spacer(1, 12))

    if dados_et:
        story.append(Paragraph("1. Parâmetros Espaço-Temporais", heading_style))
        story.append(Spacer(1, 4))
        tabela_dados = [
            ["Parâmetro", "Lado Direito (R)", "Lado Esquerdo (L)"],
            ["Tempo Médio da Passada",
             f"{dados_et['t_r']:.2f} s", f"{dados_et['t_l']:.2f} s"],
            ["Comprimento Médio",
             f"{dados_et['comp_r']:.2f} m", f"{dados_et['comp_l']:.2f} m"],
            ["Velocidade Média",
             f"{dados_et['v_r']:.2f} m/s", f"{dados_et['v_l']:.2f} m/s"],
            ["Cadência Média",
             f"{dados_et['cad_r']:.1f} ppm", f"{dados_et['cad_l']:.1f} ppm"],
        ]
        t = Table(tabela_dados, colWidths=[180, 160, 160])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1F2937')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    if observacoes:
        story.append(Paragraph("2. Observações Clínicas", heading_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(observacoes.replace('\n', '<br/>'),
                               normal_style))
        story.append(Spacer(1, 12))

    if figuras_selecionadas:
        story.append(Paragraph("3. Gráficos Selecionados", heading_style))
        story.append(Spacer(1, 8))
        for titulo_grafico, fig in figuras_selecionadas:
            story.append(Paragraph(f"<b>{titulo_grafico}</b>", normal_style))
            buf = _fig_para_buffer(fig)
            story.append(RLImage(buf, width=460, height=210))
            story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ============================================================
# 9. INTERFACE STREAMLIT
# ============================================================
st.set_page_config(page_title="Running Biomechanics - OpenSim",
                   layout="wide")
st.title("🏃 Running Biomechanics: Gait, Force, Power & Symmetry")

if 'avaliacoes' not in st.session_state:
    st.session_state.avaliacoes = []


# ---------- SIDEBAR: CADASTRO DE NOVA AVALIAÇÃO ----------
st.sidebar.header("1. Adicionar Avaliação")

idx_nova = len(st.session_state.avaliacoes)
nome_av = st.sidebar.text_input("Identificador da avaliação",
                                f"Avaliação {idx_nova + 1}",
                                key=f"nome_{idx_nova}")
vel_av = st.sidebar.number_input("Velocidade da esteira (km/h)",
                                 min_value=1.0, max_value=30.0, value=10.0,
                                 step=0.5, key=f"vel_{idx_nova}")
peso_av = st.sidebar.number_input("Peso corporal (kg)",
                                  min_value=1.0, max_value=200.0, value=66.0,
                                  step=1.0, key=f"peso_{idx_nova}")
data_av = st.sidebar.date_input("Data", pd.Timestamp.now(),
                                key=f"data_{idx_nova}").strftime('%d/%m/%Y')

f_pos = st.sidebar.file_uploader("Posições / Corpos (Kinematics)",
                                 type=['sto', 'mot', 'txt'],
                                 key=f"pos_{idx_nova}")
f_ik = st.sidebar.file_uploader("Ângulos Articulares (IK)",
                                type=['sto', 'mot', 'txt'],
                                key=f"ik_{idx_nova}")
f_angvel = st.sidebar.file_uploader("Velocidades Angulares",
                                    type=['sto', 'mot', 'txt'],
                                    key=f"av_{idx_nova}")
f_musc = st.sidebar.file_uploader("Força / Ativação Muscular",
                                  type=['sto', 'mot', 'txt'],
                                  key=f"m_{idx_nova}")
f_pow = st.sidebar.file_uploader("Potência Muscular (Power)",
                                 type=['sto', 'mot', 'txt'],
                                 key=f"p_{idx_nova}")

if st.sidebar.button("➕ Processar e adicionar", type="primary"):
    if f_pos is None:
        st.sidebar.error("O arquivo de Posições é obrigatório.")
    else:
        with st.spinner(f"Processando '{nome_av}'..."):
            try:
                av = processar_avaliacao(nome_av, vel_av, peso_av, data_av,
                                         f_pos, f_ik, f_angvel, f_musc, f_pow)
                st.session_state.avaliacoes.append(av)
                st.sidebar.success(
                    f"'{nome_av}' adicionada! "
                    f"Ciclos: R={av.n_ciclos_r}, L={av.n_ciclos_l}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Erro ao processar: {e}")


# ---------- SIDEBAR: LISTA DE AVALIAÇÕES ----------
if st.session_state.avaliacoes:
    st.sidebar.markdown("---")
    st.sidebar.header(
        f"2. Avaliações carregadas ({len(st.session_state.avaliacoes)})")
    for i, av in enumerate(list(st.session_state.avaliacoes)):
        cols = st.sidebar.columns([4, 1])
        cols[0].markdown(f"**{av.nome}**  \n`{av.velocidade:.1f} km/h` | "
                         f"`{av.data}` | R={av.n_ciclos_r} L={av.n_ciclos_l}")
        if cols[1].button("🗑", key=f"del_{av.id}_{i}"):
            st.session_state.avaliacoes.pop(i)
            st.rerun()

    if st.sidebar.button("🗑 Limpar todas"):
        st.session_state.avaliacoes = []
        st.rerun()


# ---------- CORPO PRINCIPAL ----------
if not st.session_state.avaliacoes:
    st.info("👈 Cadastre a primeira avaliação na barra lateral para começar.")
    st.stop()

# ---------- CONTROLES GLOBAIS DE ENSEMBLE ----------
with st.expander("⚙️ Configurações do Ensemble Average", expanded=False):
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        metrica_ui = st.radio(
            "Métrica central",
            options=['mean_sd', 'median_iqr'],
            format_func=lambda x: {'mean_sd': 'Média ± SD (ddof=1)',
                                   'median_iqr': 'Mediana ± IQR'}[x],
            horizontal=False,
            key='cfg_metrica'
        )
    with col_cfg2:
        interp_ui = st.radio(
            "Interpolação",
            options=['linear', 'cubic'],
            format_func=lambda x: {'linear': 'Linear (np.interp)',
                                   'cubic': 'Spline cúbica'}[x],
            horizontal=False,
            key='cfg_interp'
        )
    with col_cfg3:
        suav_ui = st.checkbox(
            "Suavizar cada ciclo (Savitzky-Golay)",
            value=SUAVIZAR_CICLOS,
            key='cfg_suav'
        )

METRICA_ENSEMBLE = metrica_ui
INTERP_METHOD = interp_ui
SUAVIZAR_CICLOS = suav_ui

if st.session_state.avaliacoes:
    st.caption(
        "ℹ️ Alterações em *Métrica*, *Interpolação* e *Suavização* afetam "
        "apenas avaliações adicionadas **depois** da mudança. "
        "Para reprocessar as existentes, remova e adicione novamente."
    )

modo = st.radio("**Modo de visualização**",
                ["👤 Individual", "📊 Comparar avaliações"],
                horizontal=True)


# ============================================================
# MODO INDIVIDUAL
# ============================================================
if modo.startswith("👤"):
    nomes = [av.nome for av in st.session_state.avaliacoes]
    nome_sel = st.selectbox("Avaliação a visualizar", nomes)

    av = next((x for x in st.session_state.avaliacoes
               if x.nome == nome_sel), None)
    if av is None:
        st.error(f"Avaliação '{nome_sel}' não encontrada. "
                 f"Disponíveis: {nomes}")
        st.stop()

    # --- 1. Espaço-temporais ---
    st.header("1. Parâmetros Espaço-Temporais (Método Leitch et al. 2011)")
    st.caption(f"⚙️ Amostragem: `{av.fs:.1f} Hz` | Massa: `{av.peso:.1f} kg` | "
               f"Ciclos: `R = {av.n_ciclos_r}` | `L = {av.n_ciclos_l}`")

    if av.dados_et:
        d = av.dados_et
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Passada Média R (s)", f"{d['t_r']:.2f}s")
        c2.metric("Passada Média L (s)", f"{d['t_l']:.2f}s")
        c3.metric("Comprimento R (m)", f"{d['comp_r']:.2f}m")
        c4.metric("Comprimento L (m)", f"{d['comp_l']:.2f}m")
    else:
        st.warning("Não foi possível calcular parâmetros espaço-temporais.")

    # --- Diagnóstico de variáveis detectadas ---
    with st.expander("🔍 Variáveis detectadas nos arquivos", expanded=False):
        tipos = {}
        for k, v in av.ensembles.items():
            tipos.setdefault(v['tipo'], []).append(v['label'])
        for tipo, labels in sorted(tipos.items()):
            st.markdown(f"**{tipo}** ({len(labels)})")
            st.write(", ".join(sorted(set(labels))))

    # --- 2. Cinemática articular ---
    st.header("2. Cinemática Articular: Ensemble Average (Média ± SD)")
    chaves_ik = [k for k, v in av.ensembles.items() if v['tipo'] == 'ik']
    chaves_pelvis = [k for k, v in av.ensembles.items() if v['tipo'] == 'pelvis']
    chaves_angvel = [k for k, v in av.ensembles.items() if v['tipo'] == 'angvel']
    chaves_pelvis_angvel = [k for k, v in av.ensembles.items()
                            if v['tipo'] == 'pelvis_angvel']

    opcoes_cin = (chaves_pelvis + chaves_pelvis_angvel
                  + chaves_ik + chaves_angvel)
    if opcoes_cin:
        labels = {k: av.ensembles[k]['label'] for k in opcoes_cin}
        chave_cin = st.selectbox("Articulação / Movimento",
                                 opcoes_cin,
                                 format_func=lambda k: labels[k])
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.pyplot(plot_ensemble_individual(av, chave_cin,
                                               metrica=METRICA_ENSEMBLE))
        with col_g2:
            st.pyplot(plot_simetria(av, chave_cin, metrica=METRICA_ENSEMBLE))
    else:
        st.info("Suba os arquivos de Ângulos Articulares (IK) e/ou "
                "Velocidades Angulares para esta avaliação.")

    # --- 3. Músculos e Potência ---
    st.header("3. Atuação e Potência Muscular")
    chaves_musc = [k for k, v in av.ensembles.items() if v['tipo'] == 'musc']
    chaves_pow = [k for k, v in av.ensembles.items() if v['tipo'] == 'pow']

    if chaves_musc or chaves_pow:
        todas_musc = chaves_musc + chaves_pow
        labels_m = {k: av.ensembles[k]['label'] for k in todas_musc}
        chave_m = st.selectbox("Músculo / Grupo",
                               todas_musc,
                               format_func=lambda k: labels_m[k],
                               key="sel_musc_indiv")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.caption("**1. Força / Ativação (BW)**")
            chave_musc_equiv = chave_m.replace('pow::', 'musc::').replace(
                'powgrp::', 'muscgrp::')
            if chave_musc_equiv in av.ensembles:
                st.pyplot(plot_ensemble_individual(
                    av, chave_musc_equiv, metrica=METRICA_ENSEMBLE))
            else:
                st.info("Sem dados de força para esta variável.")

        with col_m2:
            st.caption("**2. Função de Simetria (%)**")
            if chave_m in av.ensembles:
                st.pyplot(plot_simetria(av, chave_m,
                                        metrica=METRICA_ENSEMBLE))
            elif chave_musc_equiv in av.ensembles:
                st.pyplot(plot_simetria(av, chave_musc_equiv,
                                        metrica=METRICA_ENSEMBLE))
            else:
                st.info("Sem dados de simetria.")

        with col_m3:
            st.caption("**3. Potência Muscular (W/kg)**")
            chave_pow_equiv = chave_m.replace('musc::', 'pow::').replace(
                'muscgrp::', 'powgrp::')
            if chave_pow_equiv in av.ensembles:
                st.pyplot(plot_ensemble_individual(
                    av, chave_pow_equiv, metrica=METRICA_ENSEMBLE))
            else:
                st.info("Suba o arquivo de Potência para liberar este gráfico.")
    else:
        st.info("Suba os arquivos de Força e/ou Potência para esta avaliação.")


# ============================================================
# MODO COMPARATIVO
# ============================================================
else:
    if len(st.session_state.avaliacoes) < 2:
        st.warning("Adicione pelo menos 2 avaliações para usar o modo comparativo.")
        st.stop()

    st.header("📊 Comparação entre avaliações")

    nomes = [av.nome for av in st.session_state.avaliacoes]
    selecionadas = st.multiselect(
        "Avaliações a comparar (≥ 2)",
        nomes,
        default=nomes[:min(3, len(nomes))]
    )
    avs = [av for av in st.session_state.avaliacoes if av.nome in selecionadas]

    if len(avs) < 2:
        st.warning("Selecione pelo menos 2 avaliações.")
        st.stop()

    chaves_por_av = [set(av.ensembles.keys()) for av in avs]
    chaves_comuns = sorted(set.intersection(*chaves_por_av))

    if not chaves_comuns:
        st.error("Nenhuma variável em comum entre as avaliações selecionadas. "
                 "Você precisa subir os mesmos tipos de arquivo em todas.")
        st.stop()

    labels = {}
    for av in avs:
        for k, v in av.ensembles.items():
            labels[k] = v['label']

    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        chave_cmp = st.selectbox("Variável para comparar", chaves_comuns,
                                 format_func=lambda k: f"{k} — {labels.get(k, '')}")
    with col_sel2:
        lado = st.radio("Lado exibido", ['both', 'r', 'l'], horizontal=True,
                        format_func=lambda x: {'both': 'Ambos', 'r': 'Direito',
                                               'l': 'Esquerdo'}[x])

    st.pyplot(plot_comparativo(avs, chave_cmp, lado=lado,
                               metrica=METRICA_ENSEMBLE))

    st.subheader("Parâmetros espaço-temporais")
    linhas_et = []
    for av in avs:
        d = av.dados_et
        if not d:
            continue
        linhas_et.append({
            'Avaliação': av.nome,
            'Velocidade (km/h)': av.velocidade,
            'Passada R (s)': d['t_r'],
            'Passada L (s)': d['t_l'],
            'Comp. R (m)': d['comp_r'],
            'Comp. L (m)': d['comp_l'],
            'Cadência R (ppm)': d['cad_r'],
            'Cadência L (ppm)': d['cad_l'],
        })
    if linhas_et:
        st.dataframe(pd.DataFrame(linhas_et).round(2),
                     use_container_width=True)


# ============================================================
# SEÇÃO LAUDO
# ============================================================
st.markdown("---")
st.header("4. Gerador de Laudo Biomecânico (PDF)")

col_l1, col_l2 = st.columns([1, 1])

with col_l1:
    nome_laudo = st.text_input("Nome do Avaliado / Atleta:", "Atleta Exemplo")
    obs_clinicas = st.text_area(
        "Observações / Diagnóstico Biomecânico:",
        "Atleta apresenta simetria preservada com variabilidade padrão."
    )

with col_l2:
    st.subheader("Gráficos a incluir no laudo")

    figuras_laudo: List[Tuple[str, plt.Figure]] = []

    modo_pdf = st.radio("Tipo de laudo",
                        ["Individual (uma avaliação)",
                         "Comparativo (múltiplas avaliações)"],
                        horizontal=True)

    dados_et_pdf = None

    if modo_pdf.startswith("Individual"):
        nomes_all = [av.nome for av in st.session_state.avaliacoes]
        nome_pdf = st.selectbox("Avaliação para o laudo", nomes_all,
                                key="pdf_indiv")
        av_pdf = next((x for x in st.session_state.avaliacoes
                       if x.nome == nome_pdf), None)

        if av_pdf is None:
            st.error(f"Avaliação '{nome_pdf}' não encontrada.")
        else:
            chaves_disp = list(av_pdf.ensembles.keys())
            labels_pdf = {k: av_pdf.ensembles[k]['label'] for k in chaves_disp}

            sugestao = [k for k in chaves_disp
                        if any(t in k.lower() for t in
                               ['hip_flexion', 'knee_angle', 'ankle_angle',
                                'pelvis', 'angvel',
                                'muscgrp::Quadríceps',
                                'muscgrp::Solear'])]
            escolhidas = st.multiselect(
                "Gráficos (Ensemble + Simetria de cada)",
                chaves_disp,
                default=sugestao if sugestao else chaves_disp[:min(5, len(chaves_disp))],
                format_func=lambda k: f"{k} — {labels_pdf.get(k, '')}",
            )
            for k in escolhidas:
                figuras_laudo.append((
                    f"Ensemble — {labels_pdf.get(k, k)}",
                    plot_ensemble_individual(av_pdf, k, metrica=METRICA_ENSEMBLE)
                ))
                figuras_laudo.append((
                    f"Simetria — {labels_pdf.get(k, k)}",
                    plot_simetria(av_pdf, k, metrica=METRICA_ENSEMBLE)
                ))

            dados_et_pdf = av_pdf.dados_et

    else:
        nomes_all = [av.nome for av in st.session_state.avaliacoes]
        selecionadas_pdf = st.multiselect(
            "Avaliações para o laudo comparativo",
            nomes_all,
            default=nomes_all[:min(3, len(nomes_all))],
            key="pdf_cmp",
        )
        avs_pdf = [av for av in st.session_state.avaliacoes
                   if av.nome in selecionadas_pdf]

        if len(avs_pdf) >= 2:
            chaves_por = [set(av.ensembles.keys()) for av in avs_pdf]
            comuns = sorted(set.intersection(*chaves_por)) if chaves_por else []
            labels_cmp = {}
            for av in avs_pdf:
                for k, v in av.ensembles.items():
                    labels_cmp[k] = v['label']

            escolhidas = st.multiselect(
                "Variáveis comparadas no laudo",
                comuns,
                default=comuns[:min(5, len(comuns))],
                format_func=lambda k: f"{k} — {labels_cmp.get(k, '')}",
                key="pdf_cmp_vars",
            )
            for k in escolhidas:
                figuras_laudo.append((
                    f"Comparação — {labels_cmp.get(k, k)}",
                    plot_comparativo(avs_pdf, k, metrica=METRICA_ENSEMBLE)
                ))

            dados_et_pdf = avs_pdf[0].dados_et
        else:
            st.info("Selecione pelo menos 2 avaliações.")

st.markdown("---")

if st.button("📄 Gerar Laudo PDF", type="primary"):
    if not figuras_laudo:
        st.error("Selecione ao menos uma variável para gerar o laudo.")
    else:
        with st.spinner("Gerando PDF..."):
            pdf_bytes = gerar_pdf_laudo(
                nome_laudo, obs_clinicas, dados_et_pdf, figuras_laudo
            )
        st.success("Laudo gerado!")
        st.download_button(
            label="⬇️ Baixar Laudo (PDF)",
            data=pdf_bytes,
            file_name=f"laudo_{nome_laudo.lower().replace(' ', '_')}.pdf",
            mime="application/pdf",
        )
