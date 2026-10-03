#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовое задание: аномалии потребления воды.  Запуск:  python3 analysis.py [папка_с_csv] [папка_результатов]
Все пороги вынесены в блок PARAMS в начале файла.
"""
import sys, os, numpy as np, pandas as pd
from scipy import stats

SRC = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/mnt/user-data/outputs/results/'
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- PARAMS
PARAMS = dict(
    NOISE=1e-6,          # |Δ| меньше — численный шум float (-1e-14 и т.п.), это «ноль»
    JITTER=1.0,          # падение показаний меньше 1 м³ — «дрожание», не откат счётчика
    OFFSETS=(32.768, 327.68, 3276.8),  # 2^15 * {0.001, 0.01, 0.1} — найденный артефакт регистра
    OFFSET_TOL=3.0,      # допуск при сравнении с кратным смещению, м³ (для 32.768 — 1.0)
    ANCHOR_TOL=5.0,      # выброс = показание выше «якоря» более чем на 5 м³
    MAXSPAN=120,         # макс. длина выброса (в записях)
    SPIKE_ABS=5.0,       # м³/сут — абсолютный порог всплеска
    SPIKE_REL=10.0,      # во сколько раз выше медианы положительных суток прибора
    SPIKE_REL_MIN=1.0,   # ...и не меньше м³/сут (чтобы не ловить микро-приборы)
    PHYS_D15=75.0,       # м³/сут: Qmax DN15 ≈ 3.125 м³/ч * 24
    PHYS_D20=190.0,      # м³/сут: Qmax DN20 ≈ 7.8 м³/ч * 24
    LEAK_RUN=90,         # сут подряд с расходом > 0 (по точным суткам)
    LEAK_FLOOR=0.02,     # м³/сут — «жёсткий» критерий: минимум в серии (фон ≥ 20 л/сут каждые сутки)
    LEAK_FLOOR_HI=0.10,  # м³/сут — «тяжёлый»
    ZERO_RUN=30,         # календарных суток без изменения показаний при живой связи
    ZERO_RUN_LONG=90,
    ZERO_MIN_PKT_DAYS=20,# и минимум столько суток с пакетами внутри нулевой серии
    DIED_PREV=0.05,      # м³/сут — «был живой» перед остановкой
    STEP_RATIO=3.0,      # среднее 2-й половины / 1-й (или наоборот)
    STEP_MIN_DELTA=0.1,  # м³/сут — абсолютная величина ступеньки
    STEP_MIN_DAYS=60,    # мин. покрытых суток в каждой половине
)
P = PARAMS
MAXM = {32.768: 3, 327.68: 15, 3276.8: 3}   # допустимые кратности смещений
def is_offset(x):
    """x ≈ m * B для одного из смещений B (кратность m ограничена, чтобы не ловить случайные совпадения)"""
    x = abs(x)
    for B in P['OFFSETS']:
        m = round(x / B)
        tol = 1.0 if B < 100 else P['OFFSET_TOL']
        if 1 <= m <= MAXM[B] and abs(x - m * B) <= tol:
            return True
    return False

# ----------------------------------------------------------------- LOAD
rd = lambda f, **k: pd.read_csv(os.path.join(SRC, f), sep=';', encoding='utf-8-sig', **k)
p  = rd('pokazaniya.csv', parse_dates=['data'])
pr = rd('pribory.csv')
ev = rd('sobytiya.csv', parse_dates=['data'])
p = p.sort_values(['anon_id', 'data']).reset_index(drop=True)
p['r'] = p.pokazanie.round(3)
g = p.groupby('anon_id')
p['gap'] = g.data.diff().dt.days
p['d'] = g.pokazanie.diff()
p['prev'] = g.pokazanie.shift()
p['d_clean_noise'] = np.where(p.d.abs() < P['NOISE'], 0.0, p.d)      # убрали float-шум
p['cons_raw'] = p.d_clean_noise / p.gap                                # сырой расход/сутки (для gap>1 — среднее)
pr = pr.set_index('anon_id')
limit = pr.model.fillna('').str.contains('D20').map({True: P['PHYS_D20'], False: P['PHYS_D15']})
p['phys_limit'] = p.anon_id.map(limit)

# ------------------------------------------------- 0. ОЧИСТКА: выбросы регистра и сбросы
glitch = np.zeros(len(p), bool); drop_rows = []
for a, idx in p.groupby('anon_id').indices.items():
    r = p.r.values[idx]; n = len(r); gl = np.zeros(n, bool); i = 1; drops = []
    for i in range(1, n):
        if r[i] < r[i-1] - P['JITTER']:
            j = i - 1
            while j >= 0 and r[j] > r[i] + P['ANCHOR_TOL'] and (i - j) <= P['MAXSPAN']:
                j -= 1
            anchored = j >= 0 and r[j] <= r[i] + P['ANCHOR_TOL']
            if anchored and i - 1 - j >= 1:
                gl[j+1:i] = True; drops.append([i, 'return', i-1-j, j+1])
            else:
                drops.append([i, 'nodrop_anchor', 0, j+1])
    # объединяем подряд идущие «падения без якоря» в цепочки
    k = 0; merged = []
    while k < len(drops):
        if drops[k][1] != 'nodrop_anchor': merged.append(drops[k]); k += 1; continue
        e = k
        while e + 1 < len(drops) and drops[e+1][1] == 'nodrop_anchor' and drops[e+1][0] == drops[e][0] + 1: e += 1
        i0, i1 = drops[k][0], drops[e][0]
        total = r[i0-1] - r[i1]
        if is_offset(total):   # весь ряд до падения был «завышен» на кратное смещение
            gl[drops[k][3]:i0] = True
            for q in range(k, e + 1): merged.append([drops[q][0], 'return_start', i0, 0])
        else:
            merged.append([i0, 'reset', 0, 0])
        k = e + 1
    # «хвостовой» выброс: ряд заканчивается на уровне «якорь + m·B», возврата к якорю нет.
    # (суточное значение — среднее по пакетам, поэтому ступеньки внутри суток частичные; смотрим на итоговый уровень)
    if n > 3 and not gl[n-1]:
        best_k = None
        for k in range(n - 2, max(n - 2 - P['MAXSPAN'], -1), -1):
            if gl[k]: break
            if r[n-1] - r[k] > 30 and r[k+1] - r[k] >= 15 and is_offset(r[n-1] - r[k]) and (r[k+1:] > r[k] + P['ANCHOR_TOL']).all():
                best_k = k                  # берём самый ранний подходящий якорь = начало выброса
        if best_k is not None:
            gl[best_k+1:] = True; merged.append([best_k+1, 'tail_offset', n-1-best_k, 0])
    glitch[idx] = gl
    for i, kind, span, st in merged:
        drop_rows.append((a, idx[i], kind, span))
p['glitch'] = glitch
# «постоянный» скачок: интервал с Δ, кратным смещению, показания не вернулись — исключаем только этот интервал
p['bad_int'] = (p.d >= 30) & p.d.fillna(0).map(is_offset) & ~p.glitch
D = pd.DataFrame(drop_rows, columns=['anon_id', 'row', 'kind', 'span'])
D['data'] = p.data.values[D.row]; D['prev'] = p.pokazanie.values[D.row - 1]; D['new'] = p.pokazanie.values[D.row]
D['delta'] = D.new - D.prev
D.drop(columns='row').to_csv(OUT + 'drops_classified.csv', index=False)
reset_rows = set(D.loc[D.kind == 'reset', 'row'])
p['is_reset'] = False; p.loc[list(reset_rows), 'is_reset'] = True

# ------------------------------------------------- «чистый» суточный ряд (интерполяция по календарю)
daily_parts = []
for a, gdf in p.groupby('anon_id'):
    ok = gdf[~gdf.glitch]
    if len(ok) < 2: continue
    t = ok.data.values.astype('datetime64[D]').astype(int); v = ok.pokazanie.values
    rst = (ok.is_reset | ok.bad_int).values
    days = np.arange(t[0], t[-1] + 1)
    cum = np.interp(days, t, v)
    cons = np.diff(cum)
    # интервалы между соседними чистыми точками: точные (1 сут) или размазанные
    seg = np.searchsorted(t, days[1:], side='left')   # индекс правой точки интервала
    gaplen = (t[seg] - t[seg-1])
    exact = gaplen == 1
    # интервалы со сбросом — расход неизвестен
    bad = rst[seg]
    cons = np.where(bad, np.nan, cons)
    # интервал, в котором показание упало на 1 и менее — «дрожание», расход 0
    cons = np.where((cons < 0) & ~bad, 0.0, cons)
    cons = np.where(np.abs(cons) < P['NOISE'], 0.0, cons)
    daily_parts.append(pd.DataFrame(dict(anon_id=a, data=pd.to_datetime(days[1:], unit='D'), cons=cons, exact=exact, gaplen=gaplen)))
daily = pd.concat(daily_parts, ignore_index=True)
daily.to_csv(OUT + 'daily_clean.csv.gz', index=False, compression='gzip')

def run_lengths(mask):
    """длины подряд идущих True и позиции начала"""
    res = []; s = None
    for i, m in enumerate(mask):
        if m and s is None: s = i
        if (not m) and s is not None: res.append((s, i - s)); s = None
    if s is not None: res.append((s, len(mask) - s))
    return res

summary = {}
def top(df, col, n=10, asc=False):
    return df.sort_values(col, ascending=asc).head(n)

# ================================================= 1. ОТРИЦАТЕЛЬНЫЙ РАСХОД
neg = p[p.d < 0].copy()
neg['cls'] = np.where(neg.d.abs() < P['NOISE'], '1_float_noise',
              np.where(neg.d >= -P['JITTER'], '2_jitter(<1м3)',
              np.where(neg.glitch.shift(1, fill_value=False) | neg.index.isin(D.loc[D.kind.isin(['return', 'return_start']), 'row']), '3_return_from_offset_glitch',
                       np.where(neg.is_reset, '4_true_reset', '5_other'))))
n1 = neg.groupby('cls').agg(rows=('d', 'size'), devices=('anon_id', 'nunique'), min_delta=('d', 'min'))
print('\n== 1. Отрицательные Δ (сырые) =='); print(n1)
neg.to_csv(OUT + '1_negative_rows.csv', index=False)
resets = D[D.kind == 'reset']
print('true resets:', len(resets), resets.anon_id.nunique()); print(resets.to_string(index=False))
print('devices with ANY drop>1:', D.anon_id.nunique(), ' glitch devices:', p[p.glitch].anon_id.nunique(), ' glitch rows:', int(p.glitch.sum()))
print(D.kind.value_counts())

# ================================================= 2. ВСПЛЕСКИ / 3. НЕРЕАЛЬНЫЕ
ex = p[(p.gap == 1) & (p.d_clean_noise > 0)].copy()                    # точные сутки, сырой расход
# a) сырые
raw_abs = ex[ex.cons_raw >= P['SPIKE_ABS']]
print('\n== 2. Всплески =='); print('сырые, ≥%.0f м3/сут: строк %d, приборов %d' % (P['SPIKE_ABS'], len(raw_abs), raw_abs.anon_id.nunique()))
# б) в «чистых» данных
dc = daily[daily.exact & daily.cons.notna() & (daily.cons > 0)]
med = dc.groupby('anon_id').cons.median().rename('med_pos')
dc = dc.join(med, on='anon_id')
sp_abs = dc[dc.cons >= P['SPIKE_ABS']]
sp_rel = dc[(dc.cons >= P['SPIKE_REL'] * dc.med_pos) & (dc.cons >= P['SPIKE_REL_MIN'])]
print('после очистки: abs≥%.0f: строк %d, приборов %d' % (P['SPIKE_ABS'], len(sp_abs), sp_abs.anon_id.nunique()))
print('после очистки: rel≥%gх медианы и ≥%.1f м3: строк %d, приборов %d' % (P['SPIKE_REL'], P['SPIKE_REL_MIN'], len(sp_rel), sp_rel.anon_id.nunique()))
# причина сырых всплесков
def cause(row):
    if row.glitch: return 'offset_glitch(вернулся)'
    if is_offset(row.d): return 'offset-like(не вернулся/конец ряда)'
    return 'прочее'
raw_abs = raw_abs.assign(cause=raw_abs.apply(cause, axis=1))
print(raw_abs.groupby('cause').agg(rows=('d', 'size'), devices=('anon_id', 'nunique')))
raw_abs.to_csv(OUT + '2_spikes_raw_abs.csv', index=False)
sp_abs.to_csv(OUT + '2_spikes_clean_abs.csv', index=False)
sp_rel.to_csv(OUT + '2_spikes_clean_rel.csv', index=False)
print('top clean (abs):'); print(sp_abs.sort_values('cons', ascending=False).head(12)[['anon_id', 'data', 'cons', 'med_pos']].to_string(index=False))
print('top clean (rel):'); print(sp_rel.assign(x=lambda d: d.cons / d.med_pos).sort_values('x', ascending=False).head(12)[['anon_id', 'data', 'cons', 'med_pos', 'x']].to_string(index=False))

# 3. нереальные
allint = p[(p.d_clean_noise > 0)].copy()
unreal_raw = allint[allint.cons_raw >= allint.phys_limit]
unreal_val = p[p.pokazanie >= 1e5]
print('\n== 3. Нереальные =='); print('сырые cons>=Qmax·24 (%g/%g): строк %d, приборов %d' % (P['PHYS_D15'], P['PHYS_D20'], len(unreal_raw), unreal_raw.anon_id.nunique()))
print('показание >= 1e5 м3: строк %d, приборов %d' % (len(unreal_val), unreal_val.anon_id.nunique()))
unreal_raw = unreal_raw.assign(cause=unreal_raw.apply(cause, axis=1))
print(unreal_raw.groupby('cause').agg(rows=('d', 'size'), devices=('anon_id', 'nunique')))
dcl = daily[daily.cons.notna() & (daily.cons >= P['PHYS_D15'])]
print('в ЧИСТЫХ данных остаётся ≥%g м3/сут: %d строк, %d приборов' % (P['PHYS_D15'], len(dcl), dcl.anon_id.nunique()))
unreal_raw.to_csv(OUT + '3_unrealistic_raw.csv', index=False)
print(unreal_raw.sort_values('cons_raw', ascending=False).head(12)[['anon_id', 'data', 'prev', 'pokazanie', 'd', 'gap', 'cause']].to_string(index=False))
print('>1000:', unreal_raw[unreal_raw.cons_raw > 1000].anon_id.nunique(), ' >10000:', unreal_raw[unreal_raw.cons_raw > 10000].anon_id.nunique())

# ================================================= 4. УТЕЧКА
leak = []
for a, gdf in daily.groupby('anon_id'):
    ex_ = gdf.exact.values; c = gdf.cons.values
    pos = ex_ & (c > 0)                       # точные сутки с расходом > 0
    best = (0, 0)
    for s_, L_ in run_lengths(pos):
        if L_ > best[1]: best = (s_, L_)
    x = c[best[0]:best[0] + best[1]] if best[1] else np.array([np.nan])
    nex = int(ex_.sum()); nz = int((ex_ & (c == 0)).sum())
    leak.append((a, nex, nz, nz / nex if nex else np.nan, best[1], np.nanmean(x), np.nanmin(x), np.nanpercentile(x, 5),
                 gdf.data.iloc[best[0]].date() if best[1] else None))
L = pd.DataFrame(leak, columns=['anon_id', 'exact_days', 'zero_days', 'zero_share', 'longest_pos_run', 'mean_in_run', 'min_in_run', 'p5_in_run', 'run_start'])
leak_dev = L[L.longest_pos_run >= P['LEAK_RUN']]
leak_strong = leak_dev[leak_dev.min_in_run >= P['LEAK_FLOOR']]
leak_severe = leak_dev[leak_dev.min_in_run >= P['LEAK_FLOOR_HI']]
print('\n== 4. Утечка =='); print(L.longest_pos_run.describe())
for thr in (30, 60, 90, 120, 180): print('  серия >=%3d сут подряд без нулевых суток: %d приборов' % (thr, (L.longest_pos_run >= thr).sum()))
print('КРИТЕРИЙ: серия ≥%d точных суток подряд с расходом>0: %d приборов' % (P['LEAK_RUN'], len(leak_dev)))
print('  жёсткий: ...и минимум в серии ≥%.2f м3/сут (фоновый расход ≥%d л каждые сутки): %d' % (P['LEAK_FLOOR'], P['LEAK_FLOOR'] * 1000, len(leak_strong)))
print('  тяжёлый: минимум в серии ≥%.2f м3/сут: %d' % (P['LEAK_FLOOR_HI'], len(leak_severe)))
print('  ни одного нулевого дня за весь период (≥60 точных суток): %d' % ((L.exact_days >= 60) & (L.zero_days == 0)).sum())
leak_dev.sort_values('longest_pos_run', ascending=False).to_csv(OUT + '4_leak_all.csv', index=False)
leak_strong.sort_values('min_in_run', ascending=False).to_csv(OUT + '4_leak_strong.csv', index=False)
print(leak_strong.sort_values('min_in_run', ascending=False).head(12).to_string(index=False))

# ================================================= 5. НУЛЕВОЕ ДОЛГО
zr = []
for a, gdf in p[~p.glitch].groupby('anon_id'):
    d_ = gdf.d_clean_noise.values; t = gdf.data.values.astype('datetime64[D]').astype(int); pk = gdf.paketov_za_sutki.values
    cons_ = gdf.cons_raw.values
    z = np.r_[False, (d_[1:] == 0)]
    for s_, Ln in run_lengths(z):
        span = t[s_+Ln-1] - t[s_-1]            # календарных суток: от даты перед серией до последней даты серии
        pre = cons_[max(0, s_-60):s_]; pre = pre[np.isfinite(pre)]
        zr.append((a, pd.Timestamp(t[s_-1], unit='D').date(), pd.Timestamp(t[s_+Ln-1], unit='D').date(), int(span), Ln, float(pk[s_:s_+Ln].mean()),
                   gdf.pokazanie.values[s_], s_+Ln == len(d_), float(pre.mean()) if len(pre) else np.nan, int(len(pre))))
Z = pd.DataFrame(zr, columns=['anon_id', 'from', 'to', 'span_days', 'packet_days', 'mean_packets', 'value', 'until_end', 'mean_prev60', 'n_prev'])
zlive = Z[(Z.packet_days >= P['ZERO_MIN_PKT_DAYS'])]               # «живая связь»: пакеты приходили в ≥20 суток внутри серии
zbest = zlive.sort_values('span_days', ascending=False).drop_duplicates('anon_id')
print('\n== 5. Нулевое долго ==')
for thr in (14, 30, 60, 90, 180, 300): print('  span >= %3d сут (и ≥%d суток с пакетами): %d приборов' % (thr, P['ZERO_MIN_PKT_DAYS'], (zbest.span_days >= thr).sum()))
z30 = zbest[zbest.span_days >= P['ZERO_RUN']]; z90 = zbest[zbest.span_days >= P['ZERO_RUN_LONG']]
print('КРИТЕРИЙ: ≥%d сут без изменения показаний при живой связи: %d; ≥%d сут: %d; из них «до конца ряда»: %d / %d' % (P['ZERO_RUN'], len(z30), P['ZERO_RUN_LONG'], len(z90), z30.until_end.sum(), z90.until_end.sum()))
print('  пакетов/сутки на нулевых участках (≥30): медиана %.1f; ≥10 пакетов/сут: %d приборов' % (z30.mean_packets.median(), (z30.mean_packets >= 10).sum()))
# «умер»: нулевой участок до конца ряда, а до него был нормальный расход
died = z30[(z30.until_end) & (z30.mean_prev60 >= P['DIED_PREV']) & (z30.n_prev >= 30)]
print('  из них «ожидаемо живой → встал»: до остановки ≥30 сут и ср. расход ≥%.2f м3/сут, потом нули до конца ряда: %d' % (P['DIED_PREV'], len(died)))
allz = Z.groupby('anon_id').span_days.max()
whole = L.set_index('anon_id').join(p.groupby('anon_id').size().rename('rows'))
allzero = whole[(whole.zero_share >= 0.999) & (whole.exact_days >= 60)]
print('  вообще ни разу не менялись за весь ряд (≥60 точных суток, все нули): %d' % len(allzero))
print('  доля приборов, у которых ≥50%% точных суток нулевые: %.1f%%' % (100 * (whole[whole.exact_days >= 60].zero_share >= .5).mean()))
z30.to_csv(OUT + '5_zero_long.csv', index=False); died.to_csv(OUT + '5_zero_died.csv', index=False)
print(z30.sort_values('span_days', ascending=False).head(8).to_string(index=False))
print(died.sort_values('mean_prev60', ascending=False).head(10).to_string(index=False))

# ================================================= 6. СТУПЕНЬКА
st = []
for a, gdf in daily.groupby('anon_id'):
    c = gdf.cons.values; n = len(c)
    if n < 2 * P['STEP_MIN_DAYS']: continue
    h = n // 2; c1, c2 = c[:h], c[h:]
    n1_, n2_ = np.isfinite(c1).sum(), np.isfinite(c2).sum()
    if min(n1_, n2_) < P['STEP_MIN_DAYS']: continue
    m1, m2 = np.nanmean(c1), np.nanmean(c2)
    st.append((a, n, m1, m2, (m2 / m1) if m1 > 0 else np.inf if m2 > 0 else np.nan, m2 - m1, gdf.data.iloc[h].date()))
S = pd.DataFrame(st, columns=['anon_id', 'days', 'mean_1', 'mean_2', 'ratio', 'delta', 'split_date'])
up = S[(S.ratio >= P['STEP_RATIO']) & (S.delta >= P['STEP_MIN_DELTA'])]
dn = S[(S.ratio <= 1 / P['STEP_RATIO']) & (S.delta <= -P['STEP_MIN_DELTA'])]
print('\n== 6. Ступенька ==')
print('оценено приборов (≥%d сут в каждой половине): %d' % (P['STEP_MIN_DAYS'], len(S)))
print('вверх (×≥%g, Δ≥%.1f): %d;  вниз: %d' % (P['STEP_RATIO'], P['STEP_MIN_DELTA'], len(up), len(dn)))
print('вариант ×2: вверх %d, вниз %d' % (((S.ratio >= 2) & (S.delta >= 0.1)).sum(), ((S.ratio <= .5) & (S.delta <= -0.1)).sum()))
up.to_csv(OUT + '6_step_up.csv', index=False); dn.to_csv(OUT + '6_step_down.csv', index=False)
print(up.sort_values('delta', ascending=False).head(10).to_string(index=False)); print(dn.sort_values('delta').head(10).to_string(index=False))

# ================================================= 7. МАГНИТ
print('\n== 7. Магнит ==')
mg = ev[ev.sobytie == 'магнит']; mgdev = set(mg.anon_id)
print('событий магнит: %d (уникальных прибор-сутки: %d), приборов: %d из 3000 (%.1f%%)' % (len(mg), mg.drop_duplicates(['anon_id', 'data']).shape[0], len(mgdev), len(mgdev) / 30))
print(ev.groupby('sobytie').agg(rows=('data', 'size'), devices=('anon_id', 'nunique'), days=('data', 'nunique')))
pr['mag'] = pr.index.isin(mgdev)
print('доля приборов с магнитом по типу:'); print(pr.groupby('tip').mag.agg(['mean', 'sum', 'size']).round(3))
print('события магнит по месяцам:'); print(mg.groupby(mg.data.dt.to_period('M')).size().to_string())
log_tips = pr.loc[pr.mag, 'tip'].unique()
pop = pr[pr.tip.isin(log_tips)]           # только типы, у которых магнит вообще фиксируется
print('популяция сравнения (типы с логированием магнита %s): %d приборов, с магнитом %d (%.1f%%)' % (list(log_tips), len(pop), pop.mag.sum(), 100 * pop.mag.mean()))

def mh(tables):
    """Mantel–Haenszel OR и CMH p-value по списку 2x2 таблиц [[a,b],[c,d]]"""
    num = den = 0.0; sa = se = sv = 0.0
    for (a_, b_), (c_, d_) in tables:
        n_ = a_ + b_ + c_ + d_
        if n_ < 2: continue
        num += a_ * d_ / n_; den += b_ * c_ / n_
        r1, r2, c1, c2 = a_ + b_, c_ + d_, a_ + c_, b_ + d_
        sa += a_; se += r1 * c1 / n_; sv += r1 * r2 * c1 * c2 / (n_ ** 2 * (n_ - 1))
    chi = (abs(sa - se) - 0.5) ** 2 / sv if sv > 0 else np.nan
    return (num / den if den else np.nan), 1 - stats.chi2.cdf(chi, 1)

classes = {
    'нулевое ≥30 сут': set(z30.anon_id), 'нулевое ≥90 сут': set(z90.anon_id), '«встал» (died)': set(died.anon_id),
    'ступенька вниз': set(dn.anon_id), 'ступенька вверх': set(up.anon_id),
    'утечка (≥90 сут)': set(leak_dev.anon_id), 'утечка жёсткая': set(leak_strong.anon_id),
    'выброс-offset (glitch)': set(p[p.glitch].anon_id), 'всплеск чистый rel': set(sp_rel.anon_id), 'всплеск чистый abs': set(sp_abs.anon_id),
    'отриц. «дрожание»': set(neg[neg.cls == '2_jitter(<1м3)'].anon_id),
}
rowsT = []
for k, s_ in classes.items():
    sp_ = s_ & set(pop.index)
    a_ = len(sp_ & mgdev); b_ = len(sp_ - mgdev); c_ = len(set(pop.index) & mgdev - sp_); d_ = len(set(pop.index) - mgdev - sp_)
    orr, pv_ = stats.fisher_exact([[a_, b_], [c_, d_]])
    tabs = []
    for tp, sub in pop.groupby('tip'):
        ids = set(sub.index); m_ = ids & mgdev; x_ = ids & s_
        tabs.append([[len(m_ & x_), len(x_ - m_)], [len(m_ - x_), len(ids - m_ - x_)]])
    omh, pmh = mh(tabs)
    rowsT.append((k, len(s_), len(sp_), a_, 100 * a_ / len(sp_) if sp_ else np.nan, 100 * pop.mag.mean(), orr, pv_, omh, pmh))
T = pd.DataFrame(rowsT, columns=['класс', 'приборов', 'в_популяции', 'с_магнитом', '%_с_магнитом', '%_база', 'OR', 'p_fisher', 'OR_MH(по типам)', 'p_CMH'])
print(T.round(4).to_string(index=False)); T.to_csv(OUT + '7_magnet_vs_classes.csv', index=False)

# event study: расход за 14 сут до / 14 сут после первого магнита; контроль — приборы того же набора типов без магнита, случайные даты из тех же
sers = {a: g_.set_index('data').cons for a, g_ in daily.groupby('anon_id')}
def win(a, d0):
    s_ = sers.get(a)
    if s_ is None: return None
    bef = s_[d0 - pd.Timedelta(days=14): d0 - pd.Timedelta(days=1)]; aft = s_[d0: d0 + pd.Timedelta(days=13)]
    if bef.notna().sum() >= 7 and aft.notna().sum() >= 7: return bef.mean(), aft.mean()
first = mg.groupby('anon_id').data.min()
E = pd.DataFrame([(a, d0, *w_) for a, d0 in first.items() if (w_ := win(a, d0))], columns=['anon_id', 'first_magnet', 'before14', 'after14'])
rng = np.random.default_rng(0); ctrl = []
nonm = [a for a in pop.index if (a not in mgdev) and a in sers]
for a in nonm:
    d0 = pd.Timestamp(rng.choice(first.values))
    if (w_ := win(a, d0)): ctrl.append((a, d0, *w_))
C = pd.DataFrame(ctrl, columns=['anon_id', 'd0', 'before14', 'after14'])
def rep(name, X):
    w_ = stats.wilcoxon(X.after14, X.before14)
    big = X[X.before14 >= 0.05]; dropshare = (big.after14 < 0.5 * big.before14).mean()
    print('%-34s n=%3d  ср.до=%.3f  ср.после=%.3f  мед.Δ=%+.4f  Wilcoxon p=%.3g  доля «падение ≥2×»=%.1f%% (n=%d)' % (name, len(X), X.before14.mean(), X.after14.mean(), (X.after14 - X.before14).median(), w_.pvalue, 100 * dropshare, len(big)))
    return dropshare, len(big), int((big.after14 < 0.5 * big.before14).sum())
print('event study (14 сут до vs 14 сут после первого магнита):')
rM = rep('магнит, все', E); rC = rep('контроль (без магнита)', C)
print('  первый магнит до 2026-08-01:'); rep('магнит, ранние', E[E.first_magnet < '2026-08-01'])
print('  Fisher на долю «падение ≥2×»: p=%.3g' % stats.fisher_exact([[rM[2], rM[1] - rM[2]], [rC[2], rC[1] - rC[2]]])[1])
E.to_csv(OUT + '7_magnet_eventstudy.csv', index=False)
# внутри прибора: сутки С магнитом vs БЕЗ
mgd = mg.drop_duplicates(['anon_id', 'data']).assign(m=True)
dm = daily.merge(mgd[['anon_id', 'data', 'm']], on=['anon_id', 'data'], how='left'); dm['m'] = dm.m.fillna(False).astype(bool)
dm = dm[dm.anon_id.isin(mgdev) & dm.cons.notna() & dm.exact]
pv2 = dm.groupby(['anon_id', 'm']).cons.mean().unstack().dropna()
print('внутри приборов с магнитом: ср. расход в сутки С магнитом=%.3f, БЕЗ=%.3f (n=%d), Wilcoxon p=%.3g; приборов где с магнитом меньше: %d, больше: %d' % (pv2[True].mean(), pv2[False].mean(), len(pv2), stats.wilcoxon(pv2[True], pv2[False]).pvalue, (pv2[True] < pv2[False]).sum(), (pv2[True] > pv2[False]).sum()))
# «сброс» / «Холл»
dropdays = D[D.kind != 'reset'][['anon_id', 'data']]
jit = neg[neg.cls == '2_jitter(<1м3)'][['anon_id', 'data']]
for name in ('сброс', 'Холл'):
    e_ = ev[ev.sobytie == name]; s_ = set(e_.anon_id)
    near = lambda tab: sum(((tab.anon_id == r.anon_id) & ((tab.data - r.data).abs() <= pd.Timedelta(days=1))).any() for r in e_.drop_duplicates(['anon_id', 'data']).itertuples())
    print('%s: событий %d, приборов %d | совпадение(±1 сут) с выбросом-offset: %d, с «дрожанием»: %d, с true_reset: %d | приборов также в нулевых≥30: %d, в glitch: %d' %
          (name, len(e_), len(s_), near(dropdays), near(jit), near(resets[['anon_id', 'data']].assign(data=pd.to_datetime(resets.data))), len(s_ & set(z30.anon_id)), len(s_ & set(p[p.glitch].anon_id))))

# ================================================= 8. СОБСТВЕННЫЕ НАХОДКИ
print('\n== 8. Собственные находки ==')
# F1: артефакт +2^15 в регистре
ret = D[D.kind.isin(['return', 'tail_offset', 'return_start'])]
print('F1. Артефакт регистра: приборов %d (%.1f%%), записей-выбросов %d, эпизодов %d' % (p[p.glitch].anon_id.nunique(), 100 * p[p.glitch].anon_id.nunique() / 3000, int(p.glitch.sum()), len(ret)))
steps = p[(p.d > 25) & p.d.fillna(0).map(is_offset)]
print('    скачков вверх, кратных 32.768/327.68/3276.8 (±допуск): %d у %d приборов' % (len(steps), steps.anon_id.nunique()))
mult = (steps.d / 327.68).round(); print('    кратности к 327.68: ' + str(mult.value_counts().sort_index().head(8).to_dict()))
# возвращается ли показание ровно к прежнему значению?
exact_ret = 0; tot = 0
for r_ in D[D.kind == 'return'].itertuples():
    idx = r_.Index; row = p[(p.anon_id == r_.anon_id) & (p.data == r_.data)].index[0]
    start = row - r_.span - 1
    if start >= 0 and p.anon_id.values[start] == r_.anon_id:
        tot += 1; exact_ret += abs(p.pokazanie.values[row] - p.pokazanie.values[start]) < 1.0
print('    после выброса показание вернулось в пределах 1 м3 к значению ДО выброса: %d из %d эпизодов' % (exact_ret, tot))
gd = pd.DataFrame({'glitch': pr.index.isin(set(p[p.glitch].anon_id))}, index=pr.index).join(pr[['tip', 'bs', 'model']])
gd['pk'] = p.groupby('anon_id').paketov_za_sutki.median(); gd['hi_pk'] = gd.pk >= 10
print(gd.groupby('tip').glitch.agg(['sum', 'size', 'mean']).round(3).to_string())
print(gd.groupby('hi_pk').glitch.agg(['sum', 'size', 'mean']).round(3).to_string())
print('    по BS:', gd.groupby('bs').glitch.sum().to_dict())
epi = p[p.glitch].groupby(p.data.dt.to_period('M')).anon_id.nunique(); print('    приборов с выбросом по месяцам:', epi.to_dict())
# F2: события магнит — инфраструктурный артефакт
print('F2. магнит: %.0f%% событий приходится на авг–сен 2026; AQUA2 и fudan не передают магнит (0 из %d приборов)' % (100 * (mg.data >= '2026-08-01').mean(), pr.tip.isin(['AQUA2', 'aqua2set_fudan_stm']).sum()))
# F3: охват данных и «пустые» приборы
cov = p.groupby('anon_id').agg(first=('data', 'min'), last=('data', 'max'), rows=('data', 'size'))
cov['span'] = (cov['last'] - cov['first']).dt.days + 1; cov['cover'] = cov.rows / cov.span
print('F3. охват: приборов с рядом на всех 366 сут: %d; медиана суток с пакетами %d из 366; начали после 2025-10-01: %d; закончили до 2026-10-01: %d' %
      ((cov.rows == 366).sum(), cov.rows.median(), (cov['first'] > '2025-10-01').sum(), (cov['last'] < '2026-10-01').sum()))
print('    приборов, у которых после 2026-08-01 нет данных: %d; с долей покрытия <50%% своего периода: %d' % ((cov['last'] < '2026-08-01').sum(), (cov.cover < .5).sum()))
print('    приборов с ≥50%% нулевых суток: %.1f%% (ожидаемо для реальных квартир — единицы %%)' % (100 * (L[L.exact_days >= 60].zero_share >= .5).mean()))
# F4: синхронные всплески / тяжёлые потребители / сезонность
sp_dates = sp_rel.groupby('data').anon_id.nunique().sort_values(ascending=False)
print('F4. синхронные всплески (чистые rel, ≥1 м3) по датам, топ:', sp_dates.head(5).to_dict())
mean_dev = daily.groupby('anon_id').cons.mean()
print('    «тяжёлые» потребители: среднесуточный расход ≥ 3 м3/сут: %d; ≥1: %d; медиана по приборам: %.3f' % ((mean_dev >= 3).sum(), (mean_dev >= 1).sum(), mean_dev.median()))
print(mean_dev[mean_dev >= 3].sort_values(ascending=False).head(5).round(2).to_dict())
mon = daily[daily.cons.notna() & (daily.cons < 75)].groupby(daily.data.dt.to_period('M')).cons.mean().round(3); print('    средний расход по месяцам:', mon.to_dict())
wd = daily[daily.cons.notna() & daily.exact & (daily.cons < 75)].groupby(daily.data.dt.dayofweek).cons.mean().round(3); print('    по дням недели (0=пн):', wd.to_dict())

# F5: сдвиг запятой (показание ×10) — синхронное событие
x10 = p[(p.d > 0) & (p.prev > 1) & ((p.pokazanie / p.prev).between(9.5, 10.5))]
print('F5. показание выросло ровно в ~10 раз (сдвиг запятой): %d строк у %d приборов; даты: %s' % (len(x10), x10.anon_id.nunique(), x10.data.dt.date.astype(str).value_counts().to_dict()))

# ---- «сброс» и «дрожание»: связь на уровне строк
rs = ev[ev.sobytie == 'сброс'].drop_duplicates(['anon_id', 'data'])
key = set(zip(rs.anon_id, rs.data)); keys_pm = set()
for a_, d_ in key:
    for k_ in (-1, 0, 1): keys_pm.add((a_, d_ + pd.Timedelta(days=k_)))
pj = p[p.anon_id.isin(set(rs.anon_id))]
is_j = (pj.d < -P['NOISE']) & (pj.d >= -P['JITTER']); is_ev = pd.Series([(a_, d_) in keys_pm for a_, d_ in zip(pj.anon_id, pj.data)], index=pj.index)
tab = [[int((is_j & is_ev).sum()), int((is_j & ~is_ev).sum())], [int((~is_j & is_ev).sum()), int((~is_j & ~is_ev).sum())]]
print('\n«сброс»↔«дрожание» (строки приборов с событием «сброс»): таблица', tab, ' OR=%.1f, p=%.3g' % stats.fisher_exact(tab))
print('  доля «дрожаний», у которых в ±1 сут есть «сброс»: %.0f%%; для остальных строк: %.1f%%' % (100 * tab[0][0] / sum(tab[0]), 100 * tab[1][0] / sum(tab[1])))

# ================================================= ПРИОРИТЕТ ПРОВЕРКИ
epi_cnt = D[D.kind.isin(['return', 'tail_offset', 'return_start'])].groupby('anon_id').size()
PR = pd.DataFrame(index=pr.index)
PR['tip'] = pr.tip; PR['bs'] = pr.bs
PR['offset_episodes'] = epi_cnt.reindex(PR.index).fillna(0).astype(int)
PR['true_reset'] = PR.index.isin(set(resets.anon_id))
PR['spike_clean_max'] = dc.groupby('anon_id').cons.max().reindex(PR.index)
PR['spike_clean_n_abs'] = sp_abs.groupby('anon_id').size().reindex(PR.index).fillna(0).astype(int)
PR['leak_min_in_run'] = leak_dev.set_index('anon_id').min_in_run.reindex(PR.index)
PR['leak_run'] = leak_dev.set_index('anon_id').longest_pos_run.reindex(PR.index)
PR['step_ratio'] = pd.concat([up.set_index('anon_id').ratio, dn.set_index('anon_id').ratio]).reindex(PR.index)
PR['step_delta'] = pd.concat([up.set_index('anon_id').delta, dn.set_index('anon_id').delta]).reindex(PR.index)
PR['died'] = PR.index.isin(set(died.anon_id)); PR['mean_cons'] = mean_dev.reindex(PR.index)
PR['magnet_events'] = mg.groupby('anon_id').size().reindex(PR.index).fillna(0).astype(int)
sc = (PR.true_reset * 5 + (PR.offset_episodes > 0) * 2 + np.minimum(PR.offset_episodes, 10) * 0.2
      + (PR.spike_clean_n_abs > 0) * 3 + (PR.leak_min_in_run >= P['LEAK_FLOOR_HI']).fillna(False) * 4 + (PR.leak_min_in_run >= P['LEAK_FLOOR']).fillna(False) * 1
      + (PR.step_delta.abs() >= 0.3).fillna(False) * 2 + PR.died * 2 + (PR.mean_cons >= 1).fillna(False) * 2)
PR['score'] = sc
PR = PR.sort_values('score', ascending=False)
PR[PR.score > 0].to_csv(OUT + '9_priority_devices.csv')
print('\n== ПРИОРИТЕТ: приборов со score>0: %d; топ-20 ==' % (PR.score > 0).sum())
print(PR.head(20).round(3).to_string())

for nm, obj in dict(L=L, S=S, Z=Z, D=D).items():
    obj.to_csv(OUT + f'tbl_{nm}.csv', index=False)
print('\nготово ->', OUT)