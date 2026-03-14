"""
CTI External Validation & Sensitivity Analysis
================================================
Validates the Cognitive Tilt Index (CTI) against BLS employment age data
and tests sensitivity of Gf/Gc classification choices.

Setup:
  1. Download BLS Table 11b (2024) from:
     https://www.bls.gov/cps/cpsaat11b.xlsx
     Save as: cpsaat11b.xlsx (same directory as this script)

  2. Ensure Abilities.xlsx and Skills.xlsx are in the same directory
     (same files used by skill_distance_final.py)

  3. Run:
     python3 cti_validation.py

Output:
  output_figures/fig_cti_validation.png    — CTI vs Median Age scatter
  output_figures/fig_sensitivity.png       — Sensitivity analysis comparison
  Console: correlation statistics and ranking stability

Author: Yechan Kim — Georgia Institute of Technology, M.S. Analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import os, sys, warnings, re
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
OUT = 'output_figures'
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# DEFINITIONS (same as skill_distance_final.py)
# =============================================================================
MFG_PFX = ['51-','49-','53-','17-','11-','13-','43-','47-','15-','19-','29-']

# ── Three Gf/Gc definitions for sensitivity analysis ──

# BROAD (current project definition — 11 Gf / 6 Gc)
Gf_BROAD = ['Inductive Reasoning','Deductive Reasoning','Category Flexibility',
    'Flexibility of Closure','Speed of Closure','Perceptual Speed',
    'Spatial Orientation','Visualization','Memorization','Fluency of Ideas','Originality']
Gc_BROAD = ['Oral Comprehension','Written Comprehension','Oral Expression',
    'Written Expression','Problem Sensitivity','Information Ordering']

# STRICT CHC (only abilities that unambiguously map to Gf/Gc — 6 Gf / 4 Gc)
Gf_STRICT = ['Inductive Reasoning','Deductive Reasoning','Category Flexibility',
    'Flexibility of Closure','Speed of Closure','Perceptual Speed']
Gc_STRICT = ['Oral Comprehension','Written Comprehension','Oral Expression',
    'Written Expression']

# INTERMEDIATE (strict + borderline cases with strong aging evidence — 8 Gf / 6 Gc)
Gf_MID = ['Inductive Reasoning','Deductive Reasoning','Category Flexibility',
    'Flexibility of Closure','Speed of Closure','Perceptual Speed',
    'Visualization','Memorization']
Gc_MID = ['Oral Comprehension','Written Comprehension','Oral Expression',
    'Written Expression','Problem Sensitivity','Information Ordering']

KEEP_SOCS = {
    '13-1022','13-1023','13-1041','13-1051','13-1071','13-1075','13-1081','13-2011',
    '15-1231','15-1232','15-1242','15-1244','15-1251','15-1252','15-1253','15-2051',
    '17-2041','17-2061','17-2071','17-2072','17-2081','17-2111','17-2112','17-2131',
    '17-2141','17-2199','17-3012','17-3013','17-3021','17-3023','17-3024','17-3025',
    '17-3026','17-3027','17-3029','19-2031','19-2032','19-4031','19-4042','19-4099',
    '29-9011','29-9012','47-2111','47-2152','43-5032','43-5061','43-5071','43-5111',
    '49-1011','49-2092','49-2094','49-9012','49-9021','49-9041','49-9043','49-9044',
    '49-9045','49-9051','49-9071','49-9081',
    '51-1011','51-2011','51-2021','51-2022','51-2023','51-2031','51-2041','51-2051',
    '51-2061','51-2092','51-4021','51-4022','51-4023','51-4031','51-4032','51-4033',
    '51-4034','51-4035','51-4041','51-4051','51-4052','51-4061','51-4062','51-4071',
    '51-4072','51-4081','51-4111','51-4121','51-4122','51-4191','51-4192','51-4193',
    '51-4194','51-5111','51-5112','51-5113','51-6031','51-6042','51-6052','51-6061',
    '51-6062','51-6063','51-6064','51-6091','51-6093','51-7011','51-7021','51-7041',
    '51-7042','51-8013','51-8021','51-8031','51-8091','51-8092','51-8093',
    '51-9011','51-9012','51-9021','51-9022','51-9023','51-9031','51-9032','51-9041',
    '51-9051','51-9061','51-9111','51-9123','51-9124','51-9141','51-9161','51-9162',
    '51-9191','51-9192','51-9193','51-9194','51-9195','51-9196','51-9197','51-9198',
    '53-1042','53-1043','53-3032','53-7011','53-7021','53-7041','53-7051','53-7062',
    '53-7063','53-7064','53-7065','53-7072','53-7121',
}

CAT_COL = {'Shop Floor':'#003057','Maintenance':'#B3A369','Logistics':'#4E8C6F',
    'Bridge':'#8DA0B3','Engineering':'#2980B9','Office/Mgmt':'#857437',
    'IT/Computer':'#6C7EB7','Other Mfg':'#54585A'}

def get_cat(soc):
    if soc.startswith('51-'): return 'Shop Floor'
    if soc.startswith('49-') or soc.startswith('47-'): return 'Maintenance'
    if soc.startswith('53-'): return 'Logistics'
    if soc.startswith('17-') or soc.startswith('19-') or soc.startswith('29-'): return 'Engineering'
    if soc.startswith('15-'): return 'IT/Computer'
    if soc.startswith('11-') or soc.startswith('13-'): return 'Office/Mgmt'
    if soc.startswith('43-'): return 'Bridge'
    return 'Other Mfg'

# =============================================================================
# LOAD O*NET DATA
# =============================================================================
def load_onet():
    print("[1/5] Loading O*NET data...")
    for f in ['Abilities.xlsx', 'Skills.xlsx']:
        if not os.path.exists(f):
            print(f"  ERROR: {f} not found. Place it in the same directory."); sys.exit(1)
    raw = pd.concat([pd.read_excel('Abilities.xlsx'), pd.read_excel('Skills.xlsx')], ignore_index=True)
    if 'Recommend Suppress' in raw.columns: raw = raw[raw['Recommend Suppress']!='Y']
    if 'Not Relevant' in raw.columns: raw = raw[raw['Not Relevant']!='Y']
    raw['Data Value'] = pd.to_numeric(raw['Data Value'], errors='coerce')
    raw = raw.dropna(subset=['Data Value'])
    raw['SOC7'] = raw['O*NET-SOC Code'].astype(str).str.strip().str[:7]
    mfg = raw[raw['SOC7'].apply(lambda x: any(x.startswith(p) for p in MFG_PFX))].copy()
    mfg = mfg[mfg['SOC7'].isin(KEEP_SOCS)]

    im = mfg[mfg['Scale ID']=='IM'][['O*NET-SOC Code','SOC7','Element Name','Data Value']].rename(columns={'Data Value':'IM','O*NET-SOC Code':'SOC'})
    lv = mfg[mfg['Scale ID']=='LV'][['O*NET-SOC Code','Element Name','Data Value']].rename(columns={'Data Value':'LV','O*NET-SOC Code':'SOC'})
    mg = im.merge(lv, on=['SOC','Element Name'], how='inner')
    mg['C'] = mg['IM'] * mg['LV']
    pv = mg.groupby(['SOC7','Element Name'])['C'].mean().reset_index()
    df = pv.pivot(index='SOC7', columns='Element Name', values='C')
    df = df.dropna(axis=1, thresh=len(df)*0.3).dropna(axis=0, thresh=len(df.columns)*0.3)
    if df.isna().sum().sum() > 0: df = df.fillna(df.mean())
    titles = mfg.groupby('SOC7')['Title'].first().to_dict()
    print(f"  {df.shape[0]} occupations x {df.shape[1]} dimensions")
    return df, titles

# =============================================================================
# COMPUTE CTI (multiple definitions)
# =============================================================================
def compute_cti(df, gf_list, gc_list, label):
    gf_cols = [c for c in gf_list if c in df.columns]
    gc_cols = [c for c in gc_list if c in df.columns]
    gf = df[gf_cols].mean(axis=1)
    gc = df[gc_cols].mean(axis=1)
    cti = (gc - gf) / (gc + gf + 1e-10)
    print(f"  {label}: {len(gf_cols)} Gf dims, {len(gc_cols)} Gc dims")
    return cti

# =============================================================================
# LOAD BLS TABLE 11b
# =============================================================================
def load_bls():
    print("[2/5] Loading BLS Table 11b...")
    fname = 'cpsaat11b.xlsx'
    if not os.path.exists(fname):
        print(f"\n  ERROR: {fname} not found!")
        print(f"  Download from: https://www.bls.gov/cps/cpsaat11b.xlsx")
        print(f"  Save it in the same directory as this script.\n")
        sys.exit(1)

    # BLS Excel files have messy headers — read raw and parse
    raw = pd.read_excel(fname, header=None)
    print(f"  Raw shape: {raw.shape}")

    # Find the header row (contains "Total, 16 years and over" or similar)
    header_row = None
    for i in range(min(20, len(raw))):
        row_str = ' '.join(str(x) for x in raw.iloc[i].values)
        if 'total' in row_str.lower() and ('16' in row_str or 'years' in row_str.lower()):
            header_row = i
            break
    if header_row is None:
        # Try alternative: look for "Occupation" in first column
        for i in range(min(20, len(raw))):
            if 'occupation' in str(raw.iloc[i, 0]).lower():
                header_row = i
                break
    if header_row is None:
        header_row = 3  # Common default for BLS XLSX files
        print(f"  Warning: Could not auto-detect header row, using row {header_row}")

    print(f"  Header row detected at: {header_row}")

    # The data rows start after the header
    data = raw.iloc[header_row+1:].copy()
    data.columns = range(data.shape[1])

    # Column mapping: typically
    # 0=Occupation, 1=Total, 2=16-19, 3=20-24, 4=25-34, 5=35-44, 6=45-54, 7=55-64, 8=65+, 9=Median age
    # But verify by checking header row content
    header_vals = [str(x).strip() for x in raw.iloc[header_row].values]
    print(f"  Header values: {header_vals[:10]}")

    # Parse occupation names and try to extract SOC codes
    # BLS doesn't include SOC codes directly in Table 11b, so we match by title
    results = []
    for _, row in data.iterrows():
        occ = str(row[0]).strip()
        if not occ or occ == 'nan' or len(occ) < 3:
            continue

        # Clean dots and extra whitespace from BLS occupation names
        occ_clean = re.sub(r'\.{2,}', '', occ).strip()
        occ_clean = re.sub(r'\s+', ' ', occ_clean)

        # Skip section headers (all caps or very short)
        if len(occ_clean) < 5:
            continue

        # Try to get numeric values
        try:
            total = pd.to_numeric(row[1], errors='coerce')
            age_55_64 = pd.to_numeric(row[7], errors='coerce')
            age_65p = pd.to_numeric(row[8], errors='coerce')
            median_age = pd.to_numeric(row[9], errors='coerce')

            if pd.isna(total) or total < 10:  # Skip rows with too few workers
                continue

            pct_55p = 0
            if not pd.isna(age_55_64) and not pd.isna(age_65p) and total > 0:
                pct_55p = (age_55_64 + age_65p) / total * 100

            results.append({
                'bls_title': occ_clean,
                'total_employed': total,
                'pct_55plus': pct_55p,
                'median_age': median_age
            })
        except:
            continue

    bls = pd.DataFrame(results)
    print(f"  Parsed {len(bls)} occupation rows from BLS")
    return bls

# =============================================================================
# MATCH BLS → O*NET SOC CODES
# =============================================================================
def match_bls_onet(bls, titles):
    print("[3/5] Matching BLS occupations to O*NET SOC codes...")

    # Build a lookup: normalized title → SOC7
    onet_lookup = {}
    for soc, title in titles.items():
        # Normalize: lowercase, remove common suffixes
        t = title.lower().strip()
        t = re.sub(r',?\s*(all other|except .+)$', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        onet_lookup[t] = soc
        # Also add shorter versions
        parts = t.split(',')
        if len(parts) > 1:
            onet_lookup[parts[0].strip()] = soc

    matched = {}
    for _, row in bls.iterrows():
        bls_t = row['bls_title'].lower().strip()
        bls_t = re.sub(r',?\s*(all other|except .+)$', '', bls_t)
        bls_t = re.sub(r'\s+', ' ', bls_t).strip()

        best_soc = None
        best_score = 0

        for onet_t, soc in onet_lookup.items():
            # Exact match
            if bls_t == onet_t:
                best_soc = soc; best_score = 100; break
            # Containment match
            if onet_t in bls_t or bls_t in onet_t:
                score = min(len(onet_t), len(bls_t)) / max(len(onet_t), len(bls_t)) * 90
                if score > best_score:
                    best_soc = soc; best_score = score
            # Word overlap (Jaccard)
            words_bls = set(bls_t.split())
            words_onet = set(onet_t.split())
            if len(words_bls | words_onet) > 0:
                jaccard = len(words_bls & words_onet) / len(words_bls | words_onet)
                score = jaccard * 80
                if score > best_score:
                    best_soc = soc; best_score = score

        if best_soc and best_score >= 40 and best_soc not in matched:
            matched[best_soc] = {
                'bls_title': row['bls_title'],
                'total_employed': row['total_employed'],
                'pct_55plus': row['pct_55plus'],
                'median_age': row['median_age'],
                'match_score': best_score
            }

    print(f"  Matched {len(matched)} / {len(titles)} O*NET occupations to BLS data")

    # Show some matches for verification
    sample = list(matched.items())[:5]
    for soc, info in sample:
        print(f"    {soc} {titles.get(soc,'')[:35]:<35s} ↔ {info['bls_title'][:35]:<35s} (score={info['match_score']:.0f})")

    return matched

# =============================================================================
# FIGURE: CTI vs MEDIAN AGE / % 55+
# =============================================================================
def fig_validation(matched, cti, titles):
    print("[4/5] Generating validation figure...")

    socs = [s for s in matched if s in cti.index]
    if len(socs) < 10:
        print(f"  WARNING: Only {len(socs)} matched occupations. Need more for meaningful correlation.")
        print(f"  Check if BLS file was loaded correctly.")
        return

    x = np.array([cti[s] for s in socs])
    y_median = np.array([matched[s]['median_age'] for s in socs])
    y_pct55 = np.array([matched[s]['pct_55plus'] for s in socs])
    cats = [get_cat(s) for s in socs]
    colors = [CAT_COL.get(c, '#999') for c in cats]
    sizes = [max(30, min(200, matched[s]['total_employed'] / 5)) for s in socs]

    # Filter out NaN
    valid_median = ~np.isnan(y_median) & ~np.isnan(x)
    valid_pct = ~np.isnan(y_pct55) & ~np.isnan(x) & (y_pct55 > 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # ── Left: CTI vs Median Age ──
    if valid_median.sum() >= 5:
        x_m, y_m = x[valid_median], y_median[valid_median]
        c_m = [colors[i] for i in range(len(socs)) if valid_median[i]]
        s_m = [sizes[i] for i in range(len(socs)) if valid_median[i]]
        socs_m = [socs[i] for i in range(len(socs)) if valid_median[i]]

        ax1.scatter(x_m, y_m, c=c_m, s=s_m, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=3)

        # Regression line
        slope, intercept, r, p, se = stats.linregress(x_m, y_m)
        x_line = np.linspace(x_m.min(), x_m.max(), 100)
        ax1.plot(x_line, slope*x_line + intercept, 'r--', linewidth=2, alpha=0.7,
                label=f'r = {r:.3f}, p = {p:.4f}')

        # Spearman
        rho, p_rho = stats.spearmanr(x_m, y_m)
        ax1.set_title(f'CTI vs Median Age (n={len(x_m)})\n'
                      f'Pearson r={r:.3f} (p={p:.4f}) | Spearman ρ={rho:.3f} (p={p_rho:.4f})',
                      fontsize=14, fontweight='bold')

        # Label outliers
        residuals = y_m - (slope*x_m + intercept)
        for i in np.argsort(np.abs(residuals))[-8:]:
            name = titles.get(socs_m[i], socs_m[i])[:25]
            ax1.annotate(name, (x_m[i], y_m[i]), fontsize=7, alpha=0.8,
                        textcoords="offset points", xytext=(5, 5))

        ax1.legend(fontsize=12, loc='upper left')
        ax1.set_xlabel('CTI (← Gf-dependent | Gc-dependent →)', fontsize=12)
        ax1.set_ylabel('Median Age (BLS 2024)', fontsize=12)
        ax1.grid(True, alpha=0.15)

        print(f"  Median Age: Pearson r={r:.3f} (p={p:.4f}), Spearman ρ={rho:.3f} (p={p_rho:.4f}), n={len(x_m)}")
    else:
        ax1.text(0.5, 0.5, 'Insufficient median age data', ha='center', va='center',
                transform=ax1.transAxes, fontsize=14)
        print(f"  Median Age: insufficient data ({valid_median.sum()} valid rows)")

    # ── Right: CTI vs % 55+ ──
    if valid_pct.sum() >= 5:
        x_p, y_p = x[valid_pct], y_pct55[valid_pct]
        c_p = [colors[i] for i in range(len(socs)) if valid_pct[i]]
        s_p = [sizes[i] for i in range(len(socs)) if valid_pct[i]]
        socs_p = [socs[i] for i in range(len(socs)) if valid_pct[i]]

        ax2.scatter(x_p, y_p, c=c_p, s=s_p, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=3)

        slope2, intercept2, r2, p2, se2 = stats.linregress(x_p, y_p)
        x_line2 = np.linspace(x_p.min(), x_p.max(), 100)
        ax2.plot(x_line2, slope2*x_line2 + intercept2, 'r--', linewidth=2, alpha=0.7,
                label=f'r = {r2:.3f}, p = {p2:.4f}')

        rho2, p_rho2 = stats.spearmanr(x_p, y_p)
        ax2.set_title(f'CTI vs % Workers Aged 55+ (n={len(x_p)})\n'
                      f'Pearson r={r2:.3f} (p={p2:.4f}) | Spearman ρ={rho2:.3f} (p={p_rho2:.4f})',
                      fontsize=14, fontweight='bold')

        residuals2 = y_p - (slope2*x_p + intercept2)
        for i in np.argsort(np.abs(residuals2))[-8:]:
            name = titles.get(socs_p[i], socs_p[i])[:25]
            ax2.annotate(name, (x_p[i], y_p[i]), fontsize=7, alpha=0.8,
                        textcoords="offset points", xytext=(5, 5))

        ax2.legend(fontsize=12, loc='upper left')
        ax2.set_xlabel('CTI (← Gf-dependent | Gc-dependent →)', fontsize=12)
        ax2.set_ylabel('% Workers Aged 55+ (BLS 2024)', fontsize=12)
        ax2.grid(True, alpha=0.15)

        print(f"  % 55+: Pearson r={r2:.3f} (p={p2:.4f}), Spearman ρ={rho2:.3f} (p={p_rho2:.4f}), n={len(x_p)}")
    else:
        ax2.text(0.5, 0.5, 'Insufficient age distribution data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=14)
        print(f"  % 55+: insufficient data ({valid_pct.sum()} valid rows)")

    # Shared legend for categories
    cats_used = set(cats)
    handles = [mpatches.Patch(color=CAT_COL[k], label=k) for k in CAT_COL if k in cats_used]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10,
              framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('CTI External Validation — Does CTI Predict Workforce Age Structure?\n'
                 'Hypothesis: Gc-dependent jobs (positive CTI) retain older workers longer',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_cti_validation.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_cti_validation.png")

# =============================================================================
# SENSITIVITY ANALYSIS (Limitation #1)
# =============================================================================
def fig_sensitivity(df, titles):
    print("[5/5] Sensitivity analysis (Gf/Gc classification)...")

    cti_broad = compute_cti(df, Gf_BROAD, Gc_BROAD, "Broad (11/6)")
    cti_mid   = compute_cti(df, Gf_MID, Gc_MID, "Intermediate (8/6)")
    cti_strict= compute_cti(df, Gf_STRICT, Gc_STRICT, "Strict (6/4)")

    socs = df.index.tolist()

    # ── Rank correlations ──
    r_bm, _ = stats.spearmanr(cti_broad[socs], cti_mid[socs])
    r_bs, _ = stats.spearmanr(cti_broad[socs], cti_strict[socs])
    r_ms, _ = stats.spearmanr(cti_mid[socs], cti_strict[socs])
    print(f"\n  Rank correlations:")
    print(f"    Broad vs Intermediate: ρ = {r_bm:.4f}")
    print(f"    Broad vs Strict:       ρ = {r_bs:.4f}")
    print(f"    Intermediate vs Strict: ρ = {r_ms:.4f}")

    # ── Top-10 overlap ──
    for label, cti_a, cti_b, name in [
        ("Broad vs Strict", cti_broad, cti_strict, "Gc-dependent top-10"),
        ("Broad vs Strict", cti_broad, cti_strict, "Gf-dependent top-10"),
    ]:
        if "Gc" in name:
            top_a = set(cti_a[socs].sort_values(ascending=False).head(10).index)
            top_b = set(cti_b[socs].sort_values(ascending=False).head(10).index)
        else:
            top_a = set(cti_a[socs].sort_values(ascending=True).head(10).index)
            top_b = set(cti_b[socs].sort_values(ascending=True).head(10).index)
        overlap = len(top_a & top_b)
        print(f"    {name} overlap (Broad vs Strict): {overlap}/10")

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    pairs = [
        (cti_broad, cti_mid, "Broad (11/6) vs Intermediate (8/6)", r_bm),
        (cti_broad, cti_strict, "Broad (11/6) vs Strict (6/4)", r_bs),
        (cti_mid, cti_strict, "Intermediate (8/6) vs Strict (6/4)", r_ms),
    ]

    for ax, (cti_x, cti_y, title, rho) in zip(axes, pairs):
        colors = [CAT_COL.get(get_cat(s), '#999') for s in socs]
        ax.scatter([cti_x[s] for s in socs], [cti_y[s] for s in socs],
                  c=colors, s=60, alpha=0.7, edgecolors='white', linewidth=1)

        # Identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

        ax.set_title(f'{title}\nSpearman ρ = {rho:.4f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('CTI (Definition A)', fontsize=10)
        ax.set_ylabel('CTI (Definition B)', fontsize=10)
        ax.grid(True, alpha=0.15)
        ax.set_aspect('equal')

    fig.suptitle('Sensitivity Analysis — How Much Does the Gf/Gc Classification Choice Affect CTI?\n'
                 'Each dot is one occupation; high correlation = ranking is robust to classification choice',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig_sensitivity.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fig_sensitivity.png")

    return cti_broad

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("  CTI VALIDATION & SENSITIVITY ANALYSIS")
    print("  Extends: Skill Distance Cross-Training Framework")
    print("=" * 65)

    df, titles = load_onet()
    bls = load_bls()
    matched = match_bls_onet(bls, titles)
    cti_broad = fig_sensitivity(df, titles)
    fig_validation(matched, cti_broad, titles)

    print(f"\n{'=' * 65}")
    print(f"  DONE")
    print(f"  {OUT}/fig_cti_validation.png  — CTI vs BLS age data")
    print(f"  {OUT}/fig_sensitivity.png     — Gf/Gc sensitivity analysis")
    print(f"{'=' * 65}")
