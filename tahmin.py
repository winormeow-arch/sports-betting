"""Yaklaşan maçlar için model tahminlerini üretir ve site/index.html sayfasını oluşturur."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sportsbet.dataloaders import DataLoader
from sportsbet.evaluation import ClassifierBettor, derive_complementary_events, find_latest_odds_column

# ---- AYARLAR ---------------------------------------------------------------
LIGLER = ['England', 'Spain', 'Germany', 'Italy', 'France', 'Turkey', 'Netherlands', 'Portugal']
LIG_TR = {
    'England': 'İngiltere', 'Spain': 'İspanya', 'Germany': 'Almanya', 'Italy': 'İtalya',
    'France': 'Fransa', 'Turkey': 'Türkiye', 'Netherlands': 'Hollanda', 'Portugal': 'Portekiz',
}
SEZONLAR = [2023, 2024, 2025, 2026]  # sezonun bittiği yıl; devam eden sezon otomatik eklenir
MARKETLER = ['home_win', 'draw', 'away_win', 'over_2.5', 'under_2.5']
MARKET_TR = {'home_win': '1', 'draw': 'X', 'away_win': '2', 'over_2.5': '2.5 Üst', 'under_2.5': '2.5 Alt'}
# ---------------------------------------------------------------------------


def veri_yukle():
    if os.environ.get('TEST'):
        from sportsbet.sources import SampleSoccerOdds, SampleSoccerStats

        dl = DataLoader(param_grid={'league': ['England']}, stats=SampleSoccerStats(), odds=SampleSoccerOdds())
        X, Y, O = dl.extract_train_data(odds_type='market_average')
        return X, Y, O, X.tail(12), O.tail(12)
    from sportsbet.sources import FootballDataOdds, FootballDataStats

    dl = DataLoader(
        param_grid={'league': LIGLER, 'division': [1], 'year': SEZONLAR},
        stats=FootballDataStats(),
        odds=FootballDataOdds(),
    )
    X, Y, O = dl.extract_train_data(odds_type='market_average')
    X_fix, _, O_fix = dl.extract_fixtures_data()
    return X, Y, O, X_fix, O_fix


def model_kur():
    classifier = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), ['league', 'home_team', 'away_team']),
            remainder='passthrough',
        ),
        SimpleImputer(),
        MultiOutputClassifier(LogisticRegression(solver='liblinear', random_state=7)),
    )
    return ClassifierBettor(classifier, betting_markets=MARKETLER, init_cash=10000.0, stake=50.0)


def sayi(x):
    return None if x is None or pd.isna(x) else round(float(x), 2)


def tahminleri_uret():
    X, Y, O, X_fix, O_fix = veri_yukle()
    bettor = model_kur().fit(X, Y, O)
    if X_fix.empty:
        return []

    olasilik = bettor.predict_proba(X_fix)
    marketler = list(bettor.betting_markets_)

    # bet() sütunları complementary gruplar sırasıyla döner
    bet_sirasi = []
    for grup in derive_complementary_events(marketler):
        bet_sirasi += [m for m in marketler if m in grup]
    bahisler = bettor.bet(X_fix, O_fix)

    odds_cols = {m: find_latest_odds_column(list(O_fix.columns), m) for m in marketler}

    maclar = []
    for i in range(len(X_fix)):
        satir = X_fix.iloc[i]
        mac = {
            'tarih': pd.Timestamp(X_fix.index[i]).strftime('%Y-%m-%d'),
            'lig': LIG_TR.get(satir['league'], str(satir['league'])),
            'ev': str(satir['home_team']),
            'dep': str(satir['away_team']),
            'marketler': [],
        }
        for j, m in enumerate(marketler):
            p = float(olasilik[i, j])
            oran = sayi(O_fix.iloc[i][odds_cols[m]]) if odds_cols[m] else None
            deger = bool(bahisler[i, bet_sirasi.index(m)])
            mac['marketler'].append({
                'ad': MARKET_TR[m],
                'olasilik': round(p * 100, 1),
                'adil_oran': sayi(1 / p) if p > 0 else None,
                'oran': oran,
                'deger': deger,
            })
        maclar.append(mac)
    maclar.sort(key=lambda m: (m['tarih'], m['lig'], m['ev']))
    return maclar


HTML = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Maç Tahminleri</title>
<style>
:root{--bg:#f4f5f7;--kart:#fff;--yazi:#15171a;--soluk:#6b7280;--cizgi:#e5e7eb;--vurgu:#0a7d4f;--vurgu-bg:#e6f6ee;--bar:#3b82f6}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--kart:#181b21;--yazi:#eceef2;--soluk:#9aa1ad;--cizgi:#2a2f38;--vurgu:#34d399;--vurgu-bg:#10362a;--bar:#60a5fa}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--yazi);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:env(safe-area-inset-top) 0 env(safe-area-inset-bottom)}
header{padding:16px 16px 8px;max-width:760px;margin:auto}
h1{font-size:22px;margin:0 0 4px}
.not{color:var(--soluk);font-size:13px;margin:0}
.filtre{display:flex;gap:8px;flex-wrap:wrap;padding:8px 16px;max-width:760px;margin:auto;position:sticky;top:0;background:var(--bg);z-index:2}
select,label{font:inherit;background:var(--kart);color:var(--yazi);border:1px solid var(--cizgi);border-radius:10px;padding:8px 10px}
label{display:flex;align-items:center;gap:6px}
main{max-width:760px;margin:auto;padding:4px 16px 32px}
.gun{font-weight:600;color:var(--soluk);margin:18px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.04em}
.kart{background:var(--kart);border:1px solid var(--cizgi);border-radius:14px;padding:12px 14px;margin-bottom:10px}
.ust{display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.takim{font-weight:650;font-size:16px}
.lig{color:var(--soluk);font-size:12px;white-space:nowrap}
.yorum{font-size:13px;color:var(--soluk);margin:4px 0 10px}
.yorum b{color:var(--yazi)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{color:var(--soluk);font-weight:500;text-align:right;padding:3px 4px}
th:first-child,td:first-child{text-align:left}
td{padding:5px 4px;border-top:1px solid var(--cizgi);text-align:right;font-variant-numeric:tabular-nums}
.barwrap{display:flex;align-items:center;gap:6px;justify-content:flex-end}
.bar{height:6px;border-radius:3px;background:var(--bar);opacity:.8}
tr.deger td{background:var(--vurgu-bg)}
tr.deger td:first-child{color:var(--vurgu);font-weight:650}
.rozet{display:inline-block;font-size:11px;font-weight:700;color:var(--vurgu);border:1px solid var(--vurgu);border-radius:6px;padding:0 5px;margin-left:4px}
.bos{color:var(--soluk);text-align:center;padding:40px 0}
</style>
</head>
<body>
<header>
<h1>Maç Tahminleri</h1>
<p class="not">Güncelleme: __ZAMAN__ · Lojistik regresyon modeli, takım formuna bakar. Backtest'te uzun vadede yaklaşık başabaş çıktı; bilgi amaçlıdır.</p>
</header>
<div class="filtre">
<select id="lig"><option value="">Tüm ligler</option></select>
<label><input type="checkbox" id="sadeceDeger"> Sadece değerli bahisler</label>
</div>
<main id="liste"></main>
<script>
const MACLAR = __VERI__;
const ligSec = document.getElementById('lig'), degerSec = document.getElementById('sadeceDeger'), liste = document.getElementById('liste');
[...new Set(MACLAR.map(m => m.lig))].sort().forEach(l => { const o = document.createElement('option'); o.value = o.textContent = l; ligSec.appendChild(o); });
const f = x => x == null ? '–' : x.toFixed(2);
function gunAdi(t){ return new Date(t + 'T12:00:00').toLocaleDateString('tr-TR', {weekday:'long', day:'numeric', month:'long'}); }
function ciz(){
  const secili = MACLAR.filter(m => (!ligSec.value || m.lig === ligSec.value) && (!degerSec.checked || m.marketler.some(k => k.deger)));
  if (!secili.length){ liste.innerHTML = '<p class="bos">Gösterilecek maç yok.</p>'; return; }
  let html = '', sonGun = '';
  for (const m of secili){
    if (m.tarih !== sonGun){ html += `<div class="gun">${gunAdi(m.tarih)}</div>`; sonGun = m.tarih; }
    const ms = m.marketler.slice(0,3), fav = ms.reduce((a,b) => b.olasilik > a.olasilik ? b : a);
    const gol = m.marketler[3].olasilik >= 50 ? '2.5 Üst' : '2.5 Alt';
    const degerler = m.marketler.filter(k => k.deger).map(k => k.ad);
    html += `<div class="kart"><div class="ust"><span class="takim">${m.ev} – ${m.dep}</span><span class="lig">${m.lig}</span></div>
      <p class="yorum">Model: <b>${fav.ad}</b> (%${fav.olasilik}) · gol tarafı <b>${gol}</b>${degerler.length ? ' · değerli: <b>' + degerler.join(', ') + '</b>' : ''}</p>
      <table><tr><th>Bahis</th><th>Olasılık</th><th>Adil oran</th><th>Piyasa</th></tr>`;
    for (const k of m.marketler){
      html += `<tr class="${k.deger ? 'deger' : ''}"><td>${k.ad}${k.deger ? '<span class="rozet">DEĞER</span>' : ''}</td>
        <td><div class="barwrap"><div class="bar" style="width:${Math.round(k.olasilik*0.5)}px"></div>%${k.olasilik}</div></td>
        <td>${f(k.adil_oran)}</td><td>${f(k.oran)}</td></tr>`;
    }
    html += '</table></div>';
  }
  liste.innerHTML = html;
}
ligSec.onchange = degerSec.onchange = ciz;
ciz();
</script>
</body>
</html>
"""


def main():
    maclar = tahminleri_uret()
    zaman = datetime.now(timezone.utc).astimezone().strftime('%d.%m.%Y %H:%M UTC')
    out = Path('site')
    out.mkdir(exist_ok=True)
    (out / 'tahminler.json').write_text(json.dumps(maclar, ensure_ascii=False, indent=1), encoding='utf-8')
    html = HTML.replace('__ZAMAN__', zaman).replace('__VERI__', json.dumps(maclar, ensure_ascii=False))
    (out / 'index.html').write_text(html, encoding='utf-8')
    print(f'{len(maclar)} maç için tahmin yazıldı.')


if __name__ == '__main__':
    main()
