"""
Skill Distance-Based Cross-Training Framework for U.S. Manufacturing
=====================================================================
Quantifies cross-training viability between manufacturing occupations using
O*NET's 87-dimensional skill vectors (52 Abilities + 35 Skills) with weighted
Euclidean distance informed by cognitive aging literature (CHC model).

Extends Project 1: Energy Belt Workforce Aging — 5-Layer Risk Framework

Author: Yechan Kim — Georgia Institute of Technology, M.S. Analytics
Data:   O*NET Database v30.x (onetcenter.org/database.html)
Setup:  pip install pandas numpy scipy matplotlib scikit-learn openpyxl networkx

Output:
  output_figures/fig1_cti_map.png     Cognitive profile map (50 representative SOCs)
  output_figures/skill_explorer.html  Interactive cross-training search tool (offline)
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import json, os, sys, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = ['Arial','DejaVu Sans','sans-serif']
plt.rcParams['axes.unicode_minus'] = False
OUT = 'output_figures'
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# DEFINITIONS
# =============================================================================
Gf_EL = ['Inductive Reasoning','Deductive Reasoning','Category Flexibility',
    'Flexibility of Closure','Speed of Closure','Perceptual Speed',
    'Spatial Orientation','Visualization','Memorization','Fluency of Ideas','Originality']
Gc_EL = ['Oral Comprehension','Written Comprehension','Oral Expression',
    'Written Expression','Problem Sensitivity','Information Ordering']
PHYS = ['Static Strength','Explosive Strength','Dynamic Strength','Trunk Strength',
    'Stamina','Extent Flexibility','Dynamic Flexibility','Gross Body Coordination','Gross Body Equilibrium']
SENS = ['Near Vision','Far Vision','Visual Color Discrimination','Night Vision',
    'Peripheral Vision','Depth Perception','Glare Sensitivity','Hearing Sensitivity',
    'Auditory Attention','Sound Localization','Speech Recognition','Speech Clarity']
PSYCH = ['Arm-Hand Steadiness','Manual Dexterity','Finger Dexterity','Control Precision',
    'Multilimb Coordination','Response Orientation','Rate Control','Reaction Time',
    'Wrist-Finger Speed','Speed of Limb Movement']

# ── Skill-level CHC alignment (2D weighting: CHC direction × trainability tier) ──
SK_Gf = ['Critical Thinking','Complex Problem Solving','Active Learning','Learning Strategies']
SK_Gc = ['Reading Comprehension','Speaking','Writing','Active Listening',
    'Instructing','Social Perceptiveness','Negotiation','Persuasion','Service Orientation']
SK_PHYS = ['Equipment Maintenance','Repairing','Installation']

# Ability names for Ability vs Skill classification
AB_NAMES = {'Arm-Hand Steadiness','Auditory Attention','Category Flexibility',
    'Control Precision','Deductive Reasoning','Depth Perception','Dynamic Flexibility',
    'Dynamic Strength','Explosive Strength','Extent Flexibility','Far Vision',
    'Finger Dexterity','Flexibility of Closure','Fluency of Ideas','Glare Sensitivity',
    'Gross Body Coordination','Gross Body Equilibrium','Hearing Sensitivity',
    'Inductive Reasoning','Information Ordering','Manual Dexterity','Mathematical Reasoning',
    'Memorization','Multilimb Coordination','Near Vision','Night Vision','Number Facility',
    'Oral Comprehension','Oral Expression','Originality','Perceptual Speed',
    'Peripheral Vision','Problem Sensitivity','Rate Control','Reaction Time',
    'Response Orientation','Selective Attention','Sound Localization','Spatial Orientation',
    'Speech Clarity','Speech Recognition','Speed of Closure','Speed of Limb Movement',
    'Stamina','Static Strength','Time Sharing','Trunk Strength',
    'Visual Color Discrimination','Visualization','Wrist-Finger Speed',
    'Written Comprehension','Written Expression'}

MFG_PFX = ['51-','49-','53-','17-','11-','13-','43-','47-','15-','19-','29-']

# ── 50 SOCs for CTI Map: Comprehensive manufacturing coverage ──
FIG_SOCS = {
    # Shop Floor — Assemblers (5)
    '51-2092':('Team Assemblers','Shop Floor'),
    '51-2022':('Electrical Assemblers','Shop Floor'),
    '51-2031':('Engine Assemblers','Shop Floor'),
    '51-2041':('Metal Fabricators','Shop Floor'),
    '51-2011':('Aircraft Assemblers','Shop Floor'),
    # Shop Floor — Machining & Metalwork (10)
    '51-4041':('Machinists','Shop Floor'),
    '51-4121':('Welders/Cutters','Shop Floor'),
    '51-4111':('Tool & Die Makers','Shop Floor'),
    '51-4072':('Molding/Casting Oper','Shop Floor'),
    '51-4031':('Cutting/Press Oper','Shop Floor'),
    '51-4191':('Heat Treating Oper','Shop Floor'),
    '51-4122':('Welding Machine Oper','Shop Floor'),
    '51-9161':('CNC Tool Operators','Shop Floor'),
    '51-9162':('CNC Programmers','Shop Floor'),
    '51-4051':('Furnace Operators, Metal','Shop Floor'),
    # Shop Floor — Process & Chemical (4)
    '51-9011':('Chemical Equip Oper','Shop Floor'),
    '51-9023':('Mixing/Blending Oper','Shop Floor'),
    '51-8091':('Chemical Plant Oper','Shop Floor'),
    '51-9141':('Semiconductor Proc','Shop Floor'),
    # Shop Floor — QC, Surface, Packaging (4)
    '51-9061':('Quality Inspectors','Shop Floor'),
    '51-9124':('Coating/Painting Oper','Shop Floor'),
    '51-9111':('Packaging Machine Oper','Shop Floor'),
    '51-1011':('Production Supervisors','Shop Floor'),
    # Shop Floor — Energy Belt specialties (3)
    '51-9197':('Tire Builders','Shop Floor'),
    '51-6063':('Textile Weaving Oper','Shop Floor'),
    '51-9196':('Paper Goods Machine Oper','Shop Floor'),
    # Maintenance (5)
    '49-9041':('Ind Machinery Mechanics','Maintenance'),
    '49-9071':('Maintenance General','Maintenance'),
    '49-2094':('Electrical Repairers','Maintenance'),
    '49-9044':('Millwrights','Maintenance'),
    '49-9021':('HVAC Mechanics','Maintenance'),
    '49-1011':('Maintenance Supervisors','Maintenance'),
    # Logistics (5)
    '53-7062':('Material Handlers','Logistics'),
    '53-7051':('Forklift Operators','Logistics'),
    '53-7021':('Crane Operators','Logistics'),
    '53-1042':('Logistics Supervisors','Logistics'),
    '53-7063':('Machine Feeders','Logistics'),
    # Bridge (3)
    '43-5061':('Production Planners','Bridge'),
    '43-5071':('Shipping/Receiving Clerks','Bridge'),
    '43-5111':('Weighers/Measurers','Bridge'),
    # Specialists — entry/mid-level (5)
    '13-1071':('HR Specialists','Office/Mgmt'),
    '13-1075':('Labor Relations','Office/Mgmt'),
    '13-2011':('Accountants','Office/Mgmt'),
    '13-1081':('Logisticians','Office/Mgmt'),
    '13-1051':('Cost Estimators','Office/Mgmt'),
    # New production diversity (8)
    '51-7011':('Cabinetmakers','Shop Floor'),
    '51-5112':('Printing Press Oper','Shop Floor'),
    '51-6091':('Synthetic Fiber Oper','Shop Floor'),
    '51-9195':('Molders Non-Metal','Shop Floor'),
    '51-9193':('Cooling/Freezing Oper','Shop Floor'),
    '51-6093':('Upholsterers','Shop Floor'),
    '51-8013':('Power Plant Operators','Shop Floor'),
    # Engineering (7)
    '17-2112':('Industrial Engineers','Engineering'),
    '17-2141':('Mechanical Engineers','Engineering'),
    '17-2071':('Electrical Engineers','Engineering'),
    '17-2111':('Safety Engineers','Engineering'),
    '17-3026':('Ind Eng Technicians','Engineering'),
    '17-3024':('Mechatronics Technicians','Engineering'),
    '17-3029':('NDT Specialists','Engineering'),
    # R&D / Lab / EHS (4)
    '19-4031':('Chemical Technicians','Engineering'),
    '19-2031':('Chemists','Engineering'),
    '29-9011':('OHS Specialists','Engineering'),
    '29-9012':('OHS Technicians','Engineering'),
    # Plant Trades (2)
    '47-2111':('Plant Electricians','Maintenance'),
    '47-2152':('Pipefitters','Maintenance'),
    # IT / Computer (3)
    '15-1252':('Software Developers','IT/Computer'),
    '15-1244':('Network Administrators','IT/Computer'),
    '15-2051':('BI Analysts','IT/Computer'),
}

CAT_COL = {'Shop Floor':'#003057','Maintenance':'#B3A369','Logistics':'#4E8C6F',
    'Bridge':'#8DA0B3','Engineering':'#2980B9','Office/Mgmt':'#857437',
    'IT/Computer':'#6C7EB7','Other Mfg':'#54585A'}

def get_cat(soc):
    if soc.startswith('51-'): return 'Shop Floor'
    if soc.startswith('49-'): return 'Maintenance'
    if soc.startswith('47-'): return 'Maintenance'   # Plant electricians/pipefitters
    if soc.startswith('53-'): return 'Logistics'
    if soc.startswith('17-'): return 'Engineering'
    if soc.startswith('19-'): return 'Engineering'    # R&D Lab / EHS technicians
    if soc.startswith('29-'): return 'Engineering'    # OHS Specialists (EHS dept)
    if soc.startswith('15-'): return 'IT/Computer'
    if soc.startswith('11-') or soc.startswith('13-'): return 'Office/Mgmt'
    if soc.startswith('43-'): return 'Bridge'
    return 'Other Mfg'

# =============================================================================
# LOAD + COMPUTE
# =============================================================================
def load_onet(ab='Abilities.xlsx',sk='Skills.xlsx'):
    print("[1/4] Loading O*NET...")
    for p in [ab,sk]:
        if not os.path.exists(p):
            print(f"  ERROR: {p} not found"); sys.exit(1)
    raw=pd.concat([pd.read_excel(ab),pd.read_excel(sk)],ignore_index=True)
    if 'Recommend Suppress' in raw.columns: raw=raw[raw['Recommend Suppress']!='Y']
    if 'Not Relevant' in raw.columns: raw=raw[raw['Not Relevant']!='Y']
    raw['Data Value']=pd.to_numeric(raw['Data Value'],errors='coerce')
    raw=raw.dropna(subset=['Data Value'])
    raw['SOC7']=raw['O*NET-SOC Code'].astype(str).str.strip().str[:7]
    mfg=raw[raw['SOC7'].apply(lambda x:any(x.startswith(p) for p in MFG_PFX))].copy()

    # ── Energy Belt whitelist — Entry/Operational Level Focus ──
    # Managers removed: cross-training is for frontline/technical workforce
    # GA, TN, SC, NC, AL manufacturing
    KEEP_SOCS = {
        # ── SPECIALISTS (entry/mid-level, not managers) ──
        '13-1022',  # Wholesale and Retail Buyers
        '13-1023',  # Purchasing Agents
        '13-1041',  # Compliance Officers
        '13-1051',  # Cost Estimators
        '13-1071',  # Human Resources Specialists
        '13-1075',  # Labor Relations Specialists
        '13-1081',  # Logisticians
        '13-2011',  # Accountants and Auditors
        # ── IT — Plant-level ──
        '15-1231',  # Computer Network Support Specialists
        '15-1232',  # Computer User Support Specialists
        '15-1242',  # Database Administrators
        '15-1244',  # Network and Computer Systems Administrators
        '15-1251',  # Computer Programmers
        '15-1252',  # Software Developers
        '15-1253',  # Software QA Analysts and Testers
        '15-2051',  # Business Intelligence Analysts
        # ── ENGINEERING ──
        '17-2041',  # Chemical Engineers
        '17-2061',  # Computer Hardware Engineers
        '17-2071',  # Electrical Engineers
        '17-2072',  # Electronics Engineers
        '17-2081',  # Environmental Engineers
        '17-2111',  # Health and Safety Engineers (EHS)
        '17-2112',  # Industrial Engineers
        '17-2131',  # Materials Engineers
        '17-2141',  # Mechanical Engineers
        '17-2199',  # Manufacturing/Automation Engineers
        '17-3012',  # Electrical and Electronics Drafters
        '17-3013',  # Mechanical Drafters
        '17-3021',  # Aerospace Engineering Technicians
        '17-3023',  # Electrical/Electronic Engineering Technicians
        '17-3024',  # Electro-Mechanical/Mechatronics Technicians
        '17-3025',  # Environmental Engineering Technicians
        '17-3026',  # Industrial Engineering Technicians
        '17-3027',  # Mechanical Engineering Technicians
        '17-3029',  # Non-Destructive Testing Specialists
        # ── R&D / LAB (newly loaded via 19- prefix) ──
        '19-2031',  # Chemists — R&D formulation, failure analysis
        '19-2032',  # Materials Scientists — materials R&D
        '19-4031',  # Chemical Technicians — lab testing, QC chemistry
        '19-4042',  # Environmental Science Technicians — env monitoring
        '19-4099',  # Life/Physical Science Technicians — lab support
        # ── EHS SPECIALISTS (newly loaded via 29- prefix) ──
        '29-9011',  # Occupational Health and Safety Specialists
        '29-9012',  # Occupational Health and Safety Technicians
        # ── FACILITIES — Plant Electricians & Pipefitters (47-) ──
        '47-2111',  # Electricians — plant electrical, panel work
        '47-2152',  # Plumbers, Pipefitters, Steamfitters — piping, compressed air
        # ── PRODUCTION PLANNING / BRIDGE ──
        '43-5032',  # Dispatchers (internal plant logistics)
        '43-5061',  # Production, Planning, and Expediting Clerks
        '43-5071',  # Shipping, Receiving, and Inventory Clerks
        '43-5111',  # Weighers, Measurers, Checkers, Samplers
        # ── FIRST-LINE SUPERVISION (not manager level) ──
        # ── MAINTENANCE ──
        '49-1011',  # First-Line Supervisors of Mechanics/Repairers
        '49-2092',  # Electric Motor, Power Tool Repairers
        '49-2094',  # Electrical/Electronics Repairers, Commercial/Industrial
        '49-9012',  # Control and Valve Installers/Repairers
        '49-9021',  # HVAC Mechanics
        '49-9041',  # Industrial Machinery Mechanics
        '49-9043',  # Maintenance Workers, Machinery
        '49-9044',  # Millwrights
        '49-9045',  # Refractory Materials Repairers
        '49-9051',  # Electrical Power-Line Installers/Repairers
        '49-9071',  # Maintenance and Repair Workers, General
        '49-9081',  # Wind Turbine Service Technicians
        # ── PRODUCTION — Assembly ──
        '51-1011',  # First-Line Supervisors of Production Workers
        '51-2011',  # Aircraft Structure Assemblers
        '51-2021',  # Coil Winders, Tapers, Finishers
        '51-2022',  # Electrical/Electronic Equipment Assemblers
        '51-2023',  # Electromechanical Equipment Assemblers
        '51-2031',  # Engine and Other Machine Assemblers
        '51-2041',  # Structural Metal Fabricators and Fitters
        '51-2051',  # Fiberglass Laminators and Fabricators
        '51-2061',  # Timing Device Assemblers (precision/sensors)
        '51-2092',  # Team Assemblers
        # ── PRODUCTION — Machining & Metalwork ──
        '51-4021',  # Extruding/Drawing — Metal and Plastic
        '51-4022',  # Forging — Metal and Plastic
        '51-4023',  # Rolling — Metal and Plastic
        '51-4031',  # Cutting/Punching/Press — Metal and Plastic
        '51-4032',  # Drilling/Boring — Metal and Plastic
        '51-4033',  # Grinding/Polishing — Metal and Plastic
        '51-4034',  # Lathe/Turning — Metal and Plastic
        '51-4035',  # Milling/Planing — Metal and Plastic
        '51-4041',  # Machinists
        '51-4051',  # Metal-Refining Furnace Operators
        '51-4052',  # Pourers and Casters, Metal
        '51-4061',  # Model Makers, Metal and Plastic
        '51-4062',  # Patternmakers, Metal and Plastic
        '51-4071',  # Foundry Mold and Coremakers
        '51-4072',  # Molding/Casting Machine — Metal and Plastic
        '51-4081',  # Multiple Machine Tool Setters
        '51-4111',  # Tool and Die Makers
        '51-4121',  # Welders, Cutters, Solderers, Brazers
        '51-4122',  # Welding/Brazing Machine Operators
        '51-4191',  # Heat Treating — Metal and Plastic
        '51-4192',  # Layout Workers, Metal and Plastic
        '51-4193',  # Plating Machine — Metal and Plastic
        '51-4194',  # Tool Grinders, Filers, Sharpeners
        # ── PRODUCTION — Printing/Packaging (Graphic Packaging GA, Sonoco SC) ──
        '51-5111',  # Prepress Technicians
        '51-5112',  # Printing Press Operators
        '51-5113',  # Print Binding Workers
        # ── PRODUCTION — Textile ──
        '51-6031',  # Sewing Machine Operators
        '51-6042',  # Shoe Machine Operators
        '51-6052',  # Tailors, Dressmakers — Hanesbrands NC
        '51-6061',  # Textile Bleaching/Dyeing
        '51-6062',  # Textile Cutting
        '51-6063',  # Textile Knitting/Weaving
        '51-6064',  # Textile Winding/Drawing
        '51-6091',  # Extruding/Forming — Synthetic/Glass Fibers
        '51-6093',  # Upholsterers — auto seats (Lear, Adient AL/TN/SC)
        # ── PRODUCTION — Wood/Furniture (NC, GA) ──
        '51-7011',  # Cabinetmakers and Bench Carpenters
        '51-7021',  # Furniture Finishers — NC #1 furniture state
        '51-7041',  # Sawing Machine Operators, Wood
        '51-7042',  # Woodworking Machine Operators
        # ── PRODUCTION — Plant Utilities & Energy ──
        '51-8013',  # Power Plant Operators — TVA, Southern Co, Duke Energy
        '51-8021',  # Stationary Engineers and Boiler Operators
        '51-8031',  # Water/Wastewater Treatment Operators
        '51-8091',  # Chemical Plant and System Operators
        '51-8092',  # Gas Plant Operators
        '51-8093',  # Petroleum/Refinery Operators — Eastman TN process ops
        # ── PRODUCTION — Process/QC/Surface/Specialty ──
        '51-9011',  # Chemical Equipment Operators
        '51-9012',  # Separating/Filtering Machine
        '51-9021',  # Crushing/Grinding/Polishing Machine
        '51-9022',  # Grinding and Polishing Workers, Hand
        '51-9023',  # Mixing and Blending Machine
        '51-9031',  # Cutters and Trimmers, Hand
        '51-9032',  # Cutting and Slicing Machine
        '51-9041',  # Extruding/Forming/Pressing Machine
        '51-9051',  # Furnace/Kiln/Oven Operators
        '51-9061',  # Inspectors, Testers, Sorters, Samplers
        '51-9111',  # Packaging and Filling Machine
        '51-9123',  # Painting/Coating Workers
        '51-9124',  # Coating/Spraying Machine Operators
        '51-9141',  # Semiconductor Processing Technicians
        '51-9161',  # CNC Tool Operators
        '51-9162',  # CNC Tool Programmers
        '51-9191',  # Adhesive Bonding Machine
        '51-9192',  # Cleaning/Washing/Metal Pickling
        '51-9193',  # Cooling/Freezing Equipment Operators
        '51-9194',  # Etchers and Engravers
        '51-9195',  # Molders/Shapers/Casters (non-metal)
        '51-9196',  # Paper Goods Machine
        '51-9197',  # Tire Builders
        '51-9198',  # Helpers—Production Workers
        # ── LOGISTICS & MATERIALS ──
        '53-1042',  # Supervisors — Material Movers, Hand
        '53-1043',  # Supervisors — Material-Moving Machine Operators
        '53-3032',  # Heavy Truck Drivers
        '53-7011',  # Conveyor Operators
        '53-7021',  # Crane and Tower Operators
        '53-7041',  # Hoist and Winch Operators
        '53-7051',  # Industrial Truck/Tractor Operators (Forklift)
        '53-7062',  # Laborers/Freight/Material Movers
        '53-7063',  # Machine Feeders and Offbearers
        '53-7064',  # Packers and Packagers, Hand
        '53-7065',  # Stockers and Order Fillers
        '53-7072',  # Pump Operators
        '53-7121',  # Tank Car/Truck/Ship Loaders
    }

    before=mfg['SOC7'].nunique()
    mfg=mfg[mfg['SOC7'].isin(KEEP_SOCS)]
    after=mfg['SOC7'].nunique()
    print(f"  Energy Belt filter: {before} -> {after} SOCs")

    im=mfg[mfg['Scale ID']=='IM'][['O*NET-SOC Code','SOC7','Element Name','Data Value']].rename(columns={'Data Value':'IM','O*NET-SOC Code':'SOC'})
    lv=mfg[mfg['Scale ID']=='LV'][['O*NET-SOC Code','Element Name','Data Value']].rename(columns={'Data Value':'LV','O*NET-SOC Code':'SOC'})
    mg=im.merge(lv,on=['SOC','Element Name'],how='inner'); mg['C']=mg['IM']*mg['LV']
    pv=mg.groupby(['SOC7','Element Name'])['C'].mean().reset_index()
    df=pv.pivot(index='SOC7',columns='Element Name',values='C')
    df=df.dropna(axis=1,thresh=len(df)*0.3).dropna(axis=0,thresh=len(df.columns)*0.3)
    if df.isna().sum().sum()>0: df=df.fillna(df.mean())
    titles=mfg.groupby('SOC7')['Title'].first().to_dict()
    print(f"  {df.shape[0]} occupations x {df.shape[1]} dimensions")
    return df,titles

def compute(df):
    print("[2/4] Computing metrics...")
    dims=df.columns.tolist()
    gf_c=[c for c in Gf_EL if c in df.columns]; gc_c=[c for c in Gc_EL if c in df.columns]
    gf=df[gf_c].mean(axis=1); gc=df[gc_c].mean(axis=1); cti=(gc-gf)/(gc+gf+1e-10)
    sc=StandardScaler(); dz=pd.DataFrame(sc.fit_transform(df),index=df.index,columns=df.columns)
    # ── 2D Weighting: CHC alignment × trainability tier ──
    # Abilities (low trainability): Gf=1.5, Phys=1.2, Sens=1.1, Gc=0.8, Other=1.0
    # Skills (moderate trainability): Gf-like=1.2, Phys-like=1.1, Gc-like=0.9, Other=1.0
    w=np.ones(len(dims))
    for i,d in enumerate(dims):
        if d in AB_NAMES:  # Ability tier
            if d in Gf_EL: w[i]=1.5
            elif d in Gc_EL: w[i]=0.8
            elif d in PHYS: w[i]=1.2
            elif d in SENS: w[i]=1.1
        else:  # Skill tier
            if d in SK_Gf: w[i]=1.2
            elif d in SK_Gc: w[i]=0.9
            elif d in SK_PHYS: w[i]=1.1
    n=len(dz); dm=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            diff=dz.iloc[i].values-dz.iloc[j].values; d=np.sqrt(np.sum(w*diff**2)); dm[i,j]=d; dm[j,i]=d
    dist=pd.DataFrame(dm,index=df.index,columns=df.index)
    # Print weight summary
    ab_count=sum(1 for d in dims if d in AB_NAMES)
    sk_count=len(dims)-ab_count
    w_ne1=sum(1 for x in w if x!=1.0)
    print(f"  2D weights: {ab_count} Abilities + {sk_count} Skills, {w_ne1} non-default weights")
    return gf,gc,cti,dz,dist,w,dims

# =============================================================================
# FIG 1: CTI MAP (50 SOCs) — The sole static figure
# =============================================================================
def fig1_cti(df_all,gf,gc,cti,titles):
    print("[3/4] Fig 1: CTI Map (50 SOCs)...")
    socs=[s for s in FIG_SOCS if s in df_all.index]
    miss=[s for s in FIG_SOCS if s not in df_all.index]
    if miss: print(f"  Not in O*NET ({len(miss)}): {miss}")
    print(f"  Plotting {len(socs)} occupations")

    fig,ax=plt.subplots(figsize=(28,18))

    # Collect data for smart label placement
    pts=[(cti[s],gf[s],s) for s in socs]

    # Plot dots — larger for visibility
    for cx_,gy_,s in pts:
        cat=FIG_SOCS[s][1]; c=CAT_COL.get(cat,'#95A5A6')
        ax.scatter(cx_,gy_,c=c,s=350,alpha=0.9,edgecolors='white',linewidth=2.5,zorder=4)

    # Compute chart center for repulsion
    cx_all=[p[0] for p in pts]; cy_all=[p[1] for p in pts]
    cx_mid=(min(cx_all)+max(cx_all))/2
    cy_mid=(min(cy_all)+max(cy_all))/2

    # Smart label placement: push labels away from chart center
    import random; random.seed(42)
    label_positions=[]
    for i,(cx_,gy_,s) in enumerate(pts):
        name=FIG_SOCS[s][0]
        # Base direction: away from center
        dx=1 if cx_>=cx_mid else -1
        dy=1 if gy_>=cy_mid else -1
        # Base offset magnitude
        xoff=dx*(15+random.randint(0,10))
        yoff=dy*(10+random.randint(0,8))
        # Nudge to reduce overlap with already placed labels
        for lx,ly in label_positions:
            if abs(cx_+xoff/100-lx)<0.02 and abs(gy_+yoff/100-ly)<0.8:
                yoff+=dy*6
        label_positions.append((cx_+xoff/100, gy_+yoff/100))
        ha='left' if dx>0 else 'right'
        ax.annotate(name,(cx_,gy_),textcoords="offset points",
            xytext=(xoff,yoff),fontsize=10,fontweight='bold',color='#111',
            ha=ha,va='center',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',
                      edgecolor='#ccc',alpha=0.9),
            arrowprops=dict(arrowstyle='-',color='#999',lw=0.8,
                            connectionstyle='arc3,rad=0.1'))

    ax.axvline(x=0,color='gray',ls='--',alpha=0.35,lw=1)

    # Quadrant shading
    xl=ax.get_xlim(); yl=ax.get_ylim()
    mid_y=(yl[0]+yl[1])/2
    ax.axhspan(mid_y,yl[1],xmin=0,xmax=0.5,alpha=0.04,color='#C62828',zorder=0)
    ax.axhspan(yl[0],mid_y,xmin=0.5,xmax=1,alpha=0.04,color='#2E7D32',zorder=0)

    ax.text(xl[0]+0.005,yl[1]-0.15,
        'HIGHER RISK ZONE\nHigh Gf demand + Gf-dependent\nAdaptability required, aging vulnerability',
        fontsize=11,color='#C62828',alpha=0.6,va='top',style='italic',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='#FFF5F5',edgecolor='none',alpha=0.7))
    ax.text(xl[1]-0.005,yl[0]+0.15,
        'EXPERIENCE ZONE\nLow Gf demand + Gc-dependent\nExperience is the primary asset',
        fontsize=11,color='#2E7D32',alpha=0.6,ha='right',style='italic',
        bbox=dict(boxstyle='round,pad=0.5',facecolor='#F5FFF5',edgecolor='none',alpha=0.7))

    # Legend
    cats_used=set(FIG_SOCS[s][1] for s in socs)
    h=[mpatches.Patch(color=CAT_COL[k],label=f'{k} ({sum(1 for s in socs if FIG_SOCS[s][1]==k)})')
       for k in CAT_COL if k in cats_used]
    ax.legend(handles=h,loc='upper right',fontsize=12,framealpha=0.95,
              fancybox=True,shadow=True,title='Category (count)',title_fontsize=12)

    ax.set_xlabel('Cognitive Tilt Index (CTI)\n'
        '← Gf-dependent: Fluid intelligence, pattern recognition, adaptability    |    '
        'Gc-dependent: Crystallized intelligence, experience, accumulated knowledge →',
        fontsize=13)
    ax.set_ylabel('Gf (Fluid Intelligence) Demand Score',fontsize=14)
    ax.set_title('Cognitive Profile Map — Which Knowledge Loss is Most Critical?\n'
        f'{len(socs)} Manufacturing Occupations | O*NET 87-Dimension Analysis\n'
        'Project 1 Layer 4: Knowledge Vacuum — Gc-dependent jobs are knowledge transfer priority #1',
        fontsize=16,fontweight='bold')
    ax.grid(True,alpha=0.12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT,'fig1_cti_map.png'),dpi=200,bbox_inches='tight')
    plt.close()

    # Console summary
    gc_jobs=[(s,cti[s]) for s in socs if cti[s]>0.02]
    gf_jobs=[(s,cti[s]) for s in socs if cti[s]<-0.02]
    gc_jobs.sort(key=lambda x:x[1],reverse=True)
    gf_jobs.sort(key=lambda x:x[1])
    print(f"  Top 5 Gc-dependent (knowledge transfer priority):")
    for s,v in gc_jobs[:5]:
        print(f"    {FIG_SOCS[s][0]:<28s} CTI={v:+.3f}")
    print(f"  Top 5 Gf-dependent (highest aging risk):")
    for s,v in gf_jobs[:5]:
        print(f"    {FIG_SOCS[s][0]:<28s} CTI={v:+.3f}")
    print(f"  Saved: fig1_cti_map.png")

# =============================================================================
# INTERACTIVE HTML — Search-Based, Fully Offline
# =============================================================================
def build_html(df_all,dist_all,gf,gc,cti,titles):
    print("[4/4] Building interactive HTML (offline-compatible)...")
    pca=PCA(n_components=2); sc=StandardScaler()
    dz=pd.DataFrame(sc.fit_transform(df_all),index=df_all.index,columns=df_all.columns)
    coords=pca.fit_transform(dz)
    socs=df_all.index.tolist(); n=len(socs)
    cats=[get_cat(s) for s in socs]; tl=[titles.get(s,s) for s in socs]

    vals=dist_all.values[np.triu_indices_from(dist_all.values,k=1)]; thr=np.percentile(vals,30)
    G=nx.Graph(); G.add_nodes_from(socs)
    for i in range(n):
        for j in range(i+1,n):
            if dist_all.iloc[i,j]<thr: G.add_edge(socs[i],socs[j])
    deg=nx.degree_centrality(G); btw=nx.betweenness_centrality(G)

    nearest={}
    for s in socs:
        d=dist_all.loc[s].drop(s).sort_values().head(20)
        nearest[s]=[(idx,round(val,2)) for idx,val in zip(d.index,d.values)]

    # ── AB_NAMES defined at module level; used here for Ability vs Skill tagging ──

    # ── Z-score based distinctive dimensions (what makes each job unique) ──
    col_means = df_all.mean()
    col_stds  = df_all.std().replace(0, 1)
    zscores   = (df_all - col_means) / col_stds

    top_dims = {}
    for s in socs:
        zrow = zscores.loc[s].sort_values(ascending=False).head(8)
        top_dims[s] = [{'n': d, 'tp': 'A' if d in AB_NAMES else 'S',
                         'z': round(float(zrow[d]), 2)} for d in zrow.index]

    all_dims = sorted(df_all.columns.tolist())
    dims_json = json.dumps(all_dims)

    nodes=[{'s':s,'t':tl[i],'c':cats[i],'x':round(float(coords[i,0]),4),'y':round(float(coords[i,1]),4),
        'cti':round(float(cti.get(s,0)),3),'dc':round(float(deg.get(s,0)),3),
        'bc':round(float(btw.get(s,0)),3),'cn':int(G.degree(s)),
        'cl':CAT_COL.get(cats[i],'#78909C'),
        'td':top_dims.get(s,[]),
        'nn':[{'s':ns,'t':titles.get(ns,ns),'d':nd} for ns,nd in nearest.get(s,[])]}
        for i,s in enumerate(socs)]
    edges=[{'f':u,'t':v} for u,v in G.edges()]

    ev1=round(pca.explained_variance_ratio_[0]*100,1)
    ev2=round(pca.explained_variance_ratio_[1]*100,1)

    # ── Self-contained HTML — NO external dependencies ──
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skill Distance Explorer — U.S. Manufacturing</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,Arial,sans-serif;background:#F9F6E5}}
.hdr{{background:linear-gradient(135deg,#003057,#00467A);color:#fff;padding:18px 24px;box-shadow:0 2px 8px rgba(0,0,0,0.2);display:flex;justify-content:space-between;align-items:center}}
.hdr-left h1{{font-size:22px;margin-bottom:4px;color:#B3A369}}
.hdr-left p{{font-size:11px;opacity:0.8}}
.help-btn{{background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.3);color:#fff;width:36px;height:36px;border-radius:50%;font-size:18px;font-weight:700;cursor:pointer;transition:all 0.2s;flex-shrink:0}}
.help-btn:hover{{background:rgba(255,255,255,0.3)}}
.modal-overlay{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.5);z-index:9999;justify-content:center;align-items:center}}
.modal-overlay.show{{display:flex}}
.modal{{background:#fff;border-radius:16px;max-width:640px;width:90%;max-height:80vh;overflow-y:auto;box-shadow:0 20px 60px rgba(0,0,0,0.3);padding:32px}}
.modal h2{{font-size:20px;color:#003057;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center}}
.modal-close{{background:none;border:none;font-size:24px;cursor:pointer;color:#999;padding:4px 8px}}
.modal-close:hover{{color:#333}}
.modal h3{{font-size:14px;color:#003057;margin:16px 0 8px;border-bottom:1px solid #e8e0c8;padding-bottom:4px}}
.modal p,.modal li{{font-size:13px;color:#555;line-height:1.7}}
.modal ul{{padding-left:20px}}
.modal li{{margin-bottom:4px}}
.lang-toggle{{display:inline-flex;gap:4px;margin-bottom:16px}}
.lang-btn{{padding:6px 16px;border-radius:20px;border:1px solid #ddd;background:#f5f5f5;cursor:pointer;font-size:12px;font-weight:600;transition:all 0.15s}}
.lang-btn.active{{background:#003057;color:#fff;border-color:#003057}}
.lang-section{{display:none}}
.lang-section.show{{display:block}}
.wrap{{display:flex;height:calc(100vh - 110px)}}
.side{{width:400px;background:#fff;border-right:1px solid #ddd;display:flex;flex-direction:column}}
.search-area{{padding:16px;border-bottom:1px solid #eee}}
.sbox{{width:100%;padding:11px 14px;border:2px solid #B3A369;border-radius:10px;font-size:15px;outline:none;transition:border 0.2s}}
.sbox:focus{{border-color:#857437;box-shadow:0 0 0 3px rgba(179,163,105,0.25)}}
.rcnt{{font-size:11px;color:#999;margin-top:6px}}
.list{{flex:1;overflow-y:auto;padding:8px 16px}}
.card{{border:1px solid #e8e8e8;border-radius:10px;padding:14px;margin-bottom:8px;cursor:pointer;transition:all 0.15s}}
.card:hover{{border-color:#003057;background:#f8f5ec;transform:translateX(2px)}}
.card.act{{border-color:#B3A369;background:#f5f0e0;box-shadow:0 3px 12px rgba(179,163,105,0.2)}}
.ctt{{font-weight:700;font-size:15px;color:#222}}
.csoc{{font-size:11px;color:#888;margin-top:1px}}
.cbadge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:600;color:#fff;margin-top:5px}}
.mets{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-top:10px}}
.met{{text-align:center;padding:5px;background:#f5f6f8;border-radius:6px}}
.mv{{font-weight:700;font-size:15px;color:#333}}
.ml{{font-size:8px;color:#999;text-transform:uppercase;letter-spacing:0.5px}}
.nns{{margin-top:12px;display:none;border-top:1px solid #eee;padding-top:10px}}
.nns.show{{display:block}}
.nnt{{font-weight:700;font-size:12px;color:#003057;margin-bottom:6px}}
.nni{{display:flex;align-items:center;padding:7px 10px;background:#f8f9fa;border-radius:6px;margin-bottom:4px;font-size:12px;gap:8px}}
.nni:hover{{background:#f5f0e0}}
.nni.tgt-act{{background:#003057;color:#fff;border-radius:6px}}
.nni.tgt-act .nn-name{{color:#fff}}
.nni.tgt-act .nn-soc{{color:rgba(255,255,255,0.6)}}
.nni.tgt-act .nn-rank{{color:rgba(255,255,255,0.7)}}
.nn-rank{{font-weight:700;color:#999;width:20px;flex-shrink:0}}
.nn-info{{flex:1;min-width:0}}
.nn-name{{font-weight:600;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.nn-soc{{font-size:10px;color:#aaa}}
.nn-bar-wrap{{width:80px;flex-shrink:0}}
.nn-bar-bg{{width:100%;height:6px;background:#eee;border-radius:3px;overflow:hidden}}
.nn-bar-fill{{height:100%;border-radius:3px;transition:width 0.3s}}
.nn-dist{{font-weight:700;font-size:12px;width:40px;text-align:right;flex-shrink:0}}
.nn-more{{text-align:center;padding:8px;color:#003057;font-size:12px;font-weight:600;cursor:pointer;border-radius:6px;margin-top:4px;transition:background 0.15s}}
.nn-more:hover{{background:#F9F6E5}}
.carea{{flex:1;position:relative;background:#fff}}
canvas{{width:100%;height:100%}}
.leg{{position:absolute;top:14px;right:14px;background:rgba(255,255,255,0.97);padding:14px;border-radius:10px;border:1px solid #ddd;font-size:11px;box-shadow:0 2px 8px rgba(0,0,0,0.08)}}
.li{{display:flex;align-items:center;margin-bottom:5px;cursor:pointer;padding:3px 6px;border-radius:6px;transition:all 0.15s;user-select:none}}
.li:hover{{background:#F9F6E5}}
.li.dimmed{{opacity:0.3}}
.li .ld{{width:14px;height:14px;border-radius:50%;margin-right:8px;border:1px solid rgba(0,0,0,0.1);flex-shrink:0}}
.li .cat-count{{font-size:9px;color:#aaa;margin-left:auto;padding-left:6px}}
.ibar{{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);background:rgba(255,255,255,0.97);padding:14px 24px;border-radius:12px;border:1px solid #B3A369;font-size:14px;color:#333;box-shadow:0 4px 16px rgba(0,0,0,0.12);max-width:92%;display:flex;align-items:center;gap:20px}}
.ibar-left{{flex:1;min-width:0}}
.ibar-skills{{display:flex;flex-wrap:wrap;gap:4px;max-width:340px}}
.ibar-skill{{background:#D6DBD4;color:#54585A;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;white-space:nowrap;border:1px solid #8DA0B3}}
.ibar-skill.ab{{background:#E8D5B7;color:#6B4C1E;border-color:#C4A265}}
.ibar-skill.sk{{background:#C5D9E8;color:#1B4965;border-color:#7BA7C4}}
.zoom-ctrl{{position:absolute;bottom:80px;right:14px;display:flex;flex-direction:column;gap:4px;z-index:10}}
.zoom-btn{{width:32px;height:32px;border-radius:8px;border:1px solid #D6DBD4;background:rgba(255,255,255,0.95);color:#54585A;font-size:16px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);transition:all 0.15s}}
.zoom-btn:hover{{background:#003057;color:#fff;border-color:#003057}}
.ftr{{position:fixed;bottom:0;left:0;right:0;background:rgba(0,48,87,0.97);color:rgba(255,255,255,0.8);padding:8px 24px;font-size:11px;display:flex;justify-content:space-between;align-items:center;z-index:999}}
.ftr a{{color:#B3A369;text-decoration:none;font-weight:600;transition:opacity 0.2s}}
.ftr a:hover{{opacity:0.7}}
</style></head><body>
<div class="hdr">
<div class="hdr-left"><h1>Skill Distance Explorer</h1>
<p>Energy Belt Manufacturing (GA·TN·SC·NC·AL) | O*NET 87-D Weighted Skill Vectors | {n} Occupations</p></div>
<button class="help-btn" onclick="document.getElementById('helpModal').classList.add('show')" title="Help / 도움말">?</button>
</div>

<div class="modal-overlay" id="helpModal" onclick="if(event.target===this)this.classList.remove('show')">
<div class="modal">
<h2>How to Use / 사용법 <button class="modal-close" onclick="document.getElementById('helpModal').classList.remove('show')">&times;</button></h2>
<div class="lang-toggle">
<button class="lang-btn active" onclick="toggleLang('en',this)">English</button>
<button class="lang-btn" onclick="toggleLang('kr',this)">한국어</button>
</div>
<div class="lang-section show" id="lang-en">
<h3>Filter by Category</h3>
<p>Click any category in the legend (upper right) to <strong>hide</strong> that group from the chart and sidebar. Click again to show it. Click <strong>"Show All"</strong> to reset. This lets you focus on specific occupation groups (e.g. show only Shop Floor + Maintenance to explore internal rotation options).</p>
<h3>Search</h3>
<p>Type any occupation name, SOC code, or category (e.g. "Welder", "51-4121", "Maintenance") in the search bar. The list and chart will filter in real time.</p>
<h3>Explore a Job</h3>
<p>Click any occupation card in the sidebar. The chart will highlight that job and its nearest cross-training targets. Other nodes fade out for clarity.</p>
<h3>Cross-Training Targets</h3>
<ul>
<li>Top 5 are shown by default with distance bars</li>
<li>Click <strong>"Show 10 more"</strong> to expand to 15</li>
<li>Click <strong>"Show all"</strong> to see all 20 nearest occupations</li>
<li><strong>Click any target item</strong> to draw a comparison line on the chart showing exact distance</li>
<li><strong>Green bar</strong> = short distance (easy transition)</li>
<li><strong>Red bar</strong> = long distance (hard transition)</li>
</ul>
<h3>Reading the Chart</h3>
<ul>
<li><strong>Node size</strong> = Degree Centrality (larger = more cross-training options = career hub)</li>
<li><strong>Proximity</strong> = skill similarity (close nodes can cross-train effectively)</li>
<li>PC1 axis: Physical/Psychomotor (left) vs Cognitive/Social (right)</li>
</ul>
<h3>Key Metrics</h3>
<ul>
<li><strong>CTI</strong> (Cognitive Tilt Index): Measures whether a job relies more on experience (Gc, positive) or adaptability (Gf, negative).<br>
<em>Example: Production Planners CTI = +0.30 (heavily experience-dependent) vs CNC Operators CTI = -0.15 (adaptability-driven)</em></li>
<li><strong>Degree</strong>: How many other jobs are within cross-training range. Higher = more career mobility options.<br>
<em>Example: Machinists (Degree 0.59) are a "career hub" with many viable transitions, while Semiconductor Technicians (Degree 0.12) are highly specialized with fewer options.</em></li>
<li><strong>Distance</strong>: The 87-dimensional weighted Euclidean distance between two occupations. Lower = easier transition.<br>
<em>Example: Welder &rarr; Machinist (d=5.2, green) is highly feasible. Welder &rarr; Software Developer (d=14.8, red) requires major retraining.</em></li>
</ul>
<h3>Distance Interpretation</h3>
<ul>
<li><strong>Below 6</strong>: Highly feasible — minimal additional training needed (weeks)</li>
<li><strong>6 &ndash; 10</strong>: Moderate — structured cross-training program required (months)</li>
<li><strong>Above 10</strong>: Significant investment — formal education or apprenticeship likely needed</li>
</ul>
<h3>Why Some Office Roles Appear Close to Each Other</h3>
<p>Specialist roles like HR, Logistics, and Compliance may appear close together. O*NET measures <em>cognitive abilities</em> (reasoning, communication, decision-making), not domain knowledge. Roles that share similar ability profiles appear nearby even if their daily work differs. This actually validates cross-functional training programs common in manufacturing.</p>
</div>
<div class="lang-section" id="lang-kr">
<h3>카테고리 필터</h3>
<p>우측 상단 범례에서 카테고리를 <strong>클릭</strong>하면 해당 그룹이 차트와 사이드바에서 숨겨집니다. 다시 클릭하면 표시됩니다. <strong>"Show All"</strong>을 클릭하면 초기화됩니다. 특정 직무군만 보고 싶을 때 사용하세요 (예: Shop Floor + Maintenance만 표시하여 내부 로테이션 옵션 탐색).</p>
<h3>검색</h3>
<p>검색창에 직업명, SOC 코드, 또는 카테고리를 입력하세요 (예: "Welder", "51-4121", "Maintenance"). 목록과 차트가 실시간으로 필터링됩니다.</p>
<h3>직무 탐색</h3>
<p>사이드바에서 직업 카드를 클릭하세요. 차트에서 해당 직무와 가장 가까운 교차훈련 대상이 하이라이트되고, 나머지는 흐려집니다.</p>
<h3>교차훈련 대상</h3>
<ul>
<li>기본적으로 상위 5개가 거리 바와 함께 표시됩니다</li>
<li><strong>"Show 10 more"</strong>를 클릭하면 15개까지 확장됩니다</li>
<li><strong>"Show all"</strong>을 클릭하면 20개 전체를 볼 수 있습니다</li>
<li><strong>대상 항목을 클릭</strong>하면 차트에 두 직무 간 거리를 비교하는 연결선이 표시됩니다</li>
<li><strong>초록 바</strong> = 짧은 거리 (전환 용이)</li>
<li><strong>빨간 바</strong> = 긴 거리 (전환 어려움)</li>
</ul>
<h3>차트 읽기</h3>
<ul>
<li><strong>노드 크기</strong> = 연결 중심성 (클수록 교차훈련 옵션이 많은 커리어 허브)</li>
<li><strong>가까운 노드</strong> = 스킬 유사성이 높아 교차훈련이 효과적</li>
<li>PC1 축: 왼쪽(신체적/심리운동) vs 오른쪽(인지적/사회적)</li>
</ul>
<h3>핵심 지표</h3>
<ul>
<li><strong>CTI</strong> (인지적 틸트 인덱스): 직무가 경험(Gc, 양수)에 의존하는지 적응력(Gf, 음수)에 의존하는지 측정합니다.<br>
<em>예시: 생산 계획 담당자 CTI = +0.30 (경험 의존도 높음) vs CNC 오퍼레이터 CTI = -0.15 (적응력 중심)</em></li>
<li><strong>Degree</strong>: 교차훈련 범위 내의 직무 수. 높을수록 커리어 이동 옵션이 많습니다.<br>
<em>예시: 기계공(Degree 0.59)은 다양한 전환이 가능한 "커리어 허브"이고, 반도체 기술자(Degree 0.12)는 전문화되어 옵션이 적습니다.</em></li>
<li><strong>Distance</strong>: 두 직무 간의 87차원 가중 유클리디안 거리. 낮을수록 전환이 쉽습니다.<br>
<em>예시: 용접공 &rarr; 기계공 (d=5.2, 녹색) 매우 실현 가능. 용접공 &rarr; 소프트웨어 개발자 (d=14.8, 빨강) 대규모 재교육 필요.</em></li>
</ul>
<h3>거리 해석 기준</h3>
<ul>
<li><strong>6 미만</strong>: 매우 실현 가능 — 최소한의 추가 훈련 (주 단위)</li>
<li><strong>6 &ndash; 10</strong>: 보통 — 체계적 교차훈련 프로그램 필요 (월 단위)</li>
<li><strong>10 초과</strong>: 상당한 투자 — 정규 교육 또는 도제 과정 필요</li>
</ul>
<h3>사무직이 가까이 모이는 이유</h3>
<p>HR, 물류, 컴플라이언스 같은 전문직이 서로 가깝게 나타날 수 있습니다. O*NET은 도메인 지식이 아닌 <em>인지적 능력</em>(추론, 커뮤니케이션, 의사결정)을 측정하기 때문입니다. 유사한 능력 프로필을 가진 직무는 도메인이 달라도 가까이 위치합니다. 이는 제조업에서 교차 기능 훈련 프로그램이 효과적인 이유를 실제로 검증합니다.</p>
</div>
</div>
</div>

<div class="wrap"><div class="side"><div class="search-area">
<div style="display:flex;gap:6px;margin-bottom:8px">
<button id="tabSearch" onclick="switchTab('search')" style="flex:1;padding:7px;border:1px solid #003057;background:#003057;color:#fff;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer">Search</button>
<button id="tabJD" onclick="switchTab('jd')" style="flex:1;padding:7px;border:1px solid #003057;background:#fff;color:#003057;border-radius:6px;font-size:11px;font-weight:700;cursor:pointer">JD Match</button>
</div>
<div id="searchMode">
<input type="text" class="sbox" id="sb" placeholder="Search: Welder, 51-4121, Maintenance..." autofocus>
<div class="rcnt" id="rc">{n} occupations</div>
</div>
<div id="jdMode" style="display:none">
<textarea id="jdText" style="width:100%;height:100px;padding:10px;border:2px solid #B3A369;border-radius:10px;font-size:12px;font-family:inherit;resize:vertical" placeholder="Paste a job description here..."></textarea>
<button onclick="matchJD()" style="width:100%;padding:8px;margin-top:6px;background:#B3A369;color:#fff;border:none;border-radius:8px;font-weight:700;font-size:12px;cursor:pointer">Find Matching Occupations</button>
<div id="jdResults" style="margin-top:8px;font-size:11px;color:#555"></div>
</div>
</div>
<div class="list" id="ol"></div></div>
<div class="carea"><canvas id="cv"></canvas>
<div class="leg" id="legend">
<div style="font-weight:700;font-size:11px;color:#003057;margin-bottom:8px">Filter by Category <span style="font-weight:400;color:#aaa">(click to toggle)</span></div>
<div class="li" data-cat="Shop Floor" onclick="toggleCat(this)"><div class="ld" style="background:#003057"></div>Shop Floor<span class="cat-count"></span></div>
<div class="li" data-cat="Maintenance" onclick="toggleCat(this)"><div class="ld" style="background:#B3A369"></div>Maintenance<span class="cat-count"></span></div>
<div class="li" data-cat="Logistics" onclick="toggleCat(this)"><div class="ld" style="background:#4E8C6F"></div>Logistics<span class="cat-count"></span></div>
<div class="li" data-cat="Bridge" onclick="toggleCat(this)"><div class="ld" style="background:#8DA0B3"></div>Bridge<span class="cat-count"></span></div>
<div class="li" data-cat="Engineering" onclick="toggleCat(this)"><div class="ld" style="background:#2980B9"></div>Engineering<span class="cat-count"></span></div>
<div class="li" data-cat="Office/Mgmt" onclick="toggleCat(this)"><div class="ld" style="background:#857437"></div>Office/Mgmt<span class="cat-count"></span></div>
<div class="li" data-cat="IT/Computer" onclick="toggleCat(this)"><div class="ld" style="background:#6C7EB7"></div>IT/Computer<span class="cat-count"></span></div>
<div style="margin-top:6px;text-align:center"><span style="font-size:9px;color:#857437;cursor:pointer;font-weight:600" onclick="resetCats()">Show All</span></div>
<div style="margin-top:6px;font-size:9px;color:#aaa">Node size = Degree Centrality<br>
PC1 ({ev1}%): Physical ↔ Cognitive<br>PC2 ({ev2}%)</div></div>
<div class="zoom-ctrl">
<button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
<button class="zoom-btn" onclick="zoomOut()" title="Zoom out">&minus;</button>
<button class="zoom-btn" onclick="zoomReset()" title="Reset view" style="font-size:11px">&#8634;</button>
</div>
<div class="ibar" id="ib"><div class="ibar-left">Search or click an occupation to explore cross-training paths</div></div>
</div></div>
<div class="ftr">
<span>&copy; 2026 Yechan Kim &mdash; Georgia Institute of Technology, M.S. Analytics</span>
<div style="display:flex;gap:16px;align-items:center">
<a href="https://venmo.com/u/Yechan-Kim-20" target="_blank" rel="noopener" style="opacity:0.6;font-size:10px">☕ Buy me a coffee</a>
<a href="https://www.linkedin.com/in/yechankim/" target="_blank" rel="noopener">LinkedIn &rarr;</a>
</div>
</div>
<script>
function toggleLang(lang,btn){{
document.querySelectorAll('.lang-btn').forEach(function(b){{b.classList.remove('active')}});
btn.classList.add('active');
document.querySelectorAll('.lang-section').forEach(function(s){{s.classList.remove('show')}});
document.getElementById('lang-'+lang).classList.add('show');
}}
function switchTab(tab){{
var sBtn=document.getElementById('tabSearch'),jBtn=document.getElementById('tabJD');
var sMode=document.getElementById('searchMode'),jMode=document.getElementById('jdMode');
if(tab==='search'){{
sBtn.style.background='#003057';sBtn.style.color='#fff';
jBtn.style.background='#fff';jBtn.style.color='#003057';
sMode.style.display='block';jMode.style.display='none';
}}else{{
jBtn.style.background='#003057';jBtn.style.color='#fff';
sBtn.style.background='#fff';sBtn.style.color='#003057';
jMode.style.display='block';sMode.style.display='none';
}}
}}
function matchJD(){{
var txt=document.getElementById('jdText').value.toLowerCase();
if(!txt)return;
var dimMatches=ALL_DIMS.filter(function(d){{return txt.indexOf(d.toLowerCase())>=0}});
var scores={{}};
N.forEach(function(n){{
var sc=0;
if(n.td)n.td.forEach(function(td){{
var tdName=td.n.toLowerCase();
dimMatches.forEach(function(dm){{
if(tdName.indexOf(dm.toLowerCase())>=0||dm.toLowerCase().indexOf(tdName)>=0)sc++;
}});
}});
var ttl=n.t.toLowerCase();txt.split(/\\s+/).forEach(function(w){{
if(w.length>3&&ttl.indexOf(w)>=0)sc+=2;
}});
scores[n.s]=sc;
}});
var ranked=N.filter(function(n){{return scores[n.s]>0}}).sort(function(a,b){{return scores[b.s]-scores[a.s]}}).slice(0,10);
var res=document.getElementById('jdResults');
if(ranked.length===0){{
res.innerHTML='<div style="color:#857437;padding:8px">No matching occupations found. Try pasting a more detailed job description with specific skills and abilities.</div>';
return;
}}
var h='<div style="color:#003057;font-weight:700;margin-bottom:6px">Top Matches ('+dimMatches.length+' skills matched):</div>';
ranked.forEach(function(n,i){{
h+='<div onclick="selectNode(N.find(function(x){{return x.s===\\''+n.s+'\\'}}));switchTab(\\'search\\')" style="padding:6px 8px;background:'+(i%2===0?'#f5f0e0':'#fff')+';border-radius:4px;margin-bottom:2px;cursor:pointer;display:flex;align-items:center;gap:6px">';
h+='<span style="font-weight:700;color:#003057;width:20px">'+(i+1)+'</span>';
h+='<span style="flex:1;font-weight:600">'+n.t+'</span>';
h+='<span style="color:#857437;font-size:10px">score:'+scores[n.s]+'</span></div>';
}});
res.innerHTML=h;
}}
var N={json.dumps(nodes,separators=(',',':'))};
var E={json.dumps(edges,separators=(',',':'))};
var ALL_DIMS={dims_json};
var maxDist=Math.max.apply(null,N.map(function(n){{return n.nn.length?n.nn[n.nn.length-1].d:1}}));
var cv=document.getElementById('cv'),cx=cv.getContext('2d'),W,H,sel=null,tgt=null;
var catFilter=new Set();
function initCatCounts(){{
var counts={{}};
N.forEach(function(n){{counts[n.c]=(counts[n.c]||0)+1}});
document.querySelectorAll('.li[data-cat]').forEach(function(el){{
var cat=el.getAttribute('data-cat');
var span=el.querySelector('.cat-count');
if(span && counts[cat]) span.textContent='('+counts[cat]+')';
}});
}}
function toggleCat(el){{
var cat=el.getAttribute('data-cat');
if(catFilter.has(cat)){{catFilter.delete(cat);el.classList.remove('dimmed')}}
else{{catFilter.add(cat);el.classList.add('dimmed')}}
sel=null;tgt=null;rl(document.getElementById('sb').value);dr();
}}
function resetCats(){{
catFilter.clear();
document.querySelectorAll('.li[data-cat]').forEach(function(el){{el.classList.remove('dimmed')}});
sel=null;tgt=null;rl(document.getElementById('sb').value);dr();
}}
function isVisible(n){{return catFilter.size===0||!catFilter.has(n.c)}}
function rz(){{var r=cv.parentElement.getBoundingClientRect();W=cv.width=r.width*2;H=cv.height=r.height*2;cv.style.width=r.width+'px';cv.style.height=r.height+'px';cx.setTransform(2,0,0,2,0,0);dr()}}
var xs=N.map(function(n){{return n.x}}),ys=N.map(function(n){{return n.y}});
var xn=Math.min.apply(null,xs),xx=Math.max.apply(null,xs),yn=Math.min.apply(null,ys),yx=Math.max.apply(null,ys),pd=70;
var zm=1,panX=0,panY=0,dragging=false,dragStartX=0,dragStartY=0,dragPanX=0,dragPanY=0;
function tx(x){{return (pd+(x-xn)/(xx-xn)*(W/2-2*pd))*zm+panX}}
function ty(y){{return (pd+(yx-y)/(yx-yn)*(H/2-2*pd))*zm+panY}}
cv.addEventListener('wheel',function(e){{
e.preventDefault();
var rect=cv.getBoundingClientRect();
var mx=(e.clientX-rect.left)*2;
var my=(e.clientY-rect.top)*2;
var oldZm=zm;
zm*=e.deltaY<0?1.15:0.87;
zm=Math.max(0.5,Math.min(zm,8));
panX=mx-(mx-panX)*zm/oldZm;
panY=my-(my-panY)*zm/oldZm;
dr()}},{{passive:false}});
cv.addEventListener('mousedown',function(e){{
dragging=true;dragStartX=e.clientX;dragStartY=e.clientY;dragPanX=panX;dragPanY=panY;cv.style.cursor='grabbing'}});
cv.addEventListener('mousemove',function(e){{
if(!dragging)return;
panX=dragPanX+(e.clientX-dragStartX)*2;
panY=dragPanY+(e.clientY-dragStartY)*2;
dr()}});
cv.addEventListener('mouseup',function(){{dragging=false;cv.style.cursor='grab'}});
cv.addEventListener('mouseleave',function(){{dragging=false;cv.style.cursor='grab'}});
cv.style.cursor='grab';
function zoomIn(){{var cx0=W/4,cy0=H/4;var oz=zm;zm=Math.min(zm*1.3,8);panX=cx0-(cx0-panX)*zm/oz;panY=cy0-(cy0-panY)*zm/oz;dr()}}
function zoomOut(){{var cx0=W/4,cy0=H/4;var oz=zm;zm=Math.max(zm*0.77,0.5);panX=cx0-(cx0-panX)*zm/oz;panY=cy0-(cy0-panY)*zm/oz;dr()}}
function zoomReset(){{zm=1;panX=0;panY=0;dr()}}
function dr(){{
cx.clearRect(0,0,W/2,H/2);
var ns=sel?new Set((N.find(function(n){{return n.s===sel}})||{{}}).nn.map(function(x){{return x.s}})):null;
E.forEach(function(e){{
var f=N.find(function(n){{return n.s===e.f}}),t=N.find(function(n){{return n.s===e.t}});
if(!f||!t)return;
if(!isVisible(f)||!isVisible(t))return;
var hi=sel&&(e.f===sel||e.t===sel);
cx.beginPath();cx.moveTo(tx(f.x),ty(f.y));cx.lineTo(tx(t.x),ty(t.y));
cx.strokeStyle=hi?'#B3A369':'#ddd';cx.lineWidth=hi?2.5:0.4;
cx.globalAlpha=sel?(hi?0.8:0.03):0.07;cx.stroke()}});
cx.globalAlpha=1;
N.forEach(function(n){{
var r=4+n.dc*35,a=0.85;
var vis=isVisible(n);
if(!vis)a=0.03;
else if(sel){{if(n.s===sel)a=1;else if(n.s===tgt)a=1;else if(ns&&ns.has(n.s))a=0.95;else a=0.06}}
cx.globalAlpha=a;cx.beginPath();cx.arc(tx(n.x),ty(n.y),r,0,Math.PI*2);
cx.fillStyle=n.cl;cx.fill();
cx.strokeStyle=n.s===sel?'#222':'rgba(255,255,255,0.8)';cx.lineWidth=n.s===sel?3:1;cx.stroke();
if(a>0.2){{cx.fillStyle='#333';cx.font=(n.s===sel?'bold 10px':'7px')+' Arial';
cx.textAlign='center';cx.fillText(n.s.substring(0,7),tx(n.x),ty(n.y)-r-4)}}}});
cx.globalAlpha=1;
if(sel&&tgt){{
var sn=N.find(function(n){{return n.s===sel}});
var tn=N.find(function(n){{return n.s===tgt}});
if(sn&&tn){{
var x1=tx(sn.x),y1=ty(sn.y),x2=tx(tn.x),y2=ty(tn.y);
cx.beginPath();cx.moveTo(x1,y1);cx.lineTo(x2,y2);
cx.strokeStyle='#003057';cx.lineWidth=3;cx.globalAlpha=0.9;
cx.setLineDash([8,4]);cx.stroke();cx.setLineDash([]);
var r1=4+sn.dc*35,r2=4+tn.dc*35;
cx.globalAlpha=1;
cx.beginPath();cx.arc(x1,y1,r1+4,0,Math.PI*2);cx.strokeStyle='#003057';cx.lineWidth=3;cx.stroke();
cx.beginPath();cx.arc(x2,y2,r2+4,0,Math.PI*2);cx.strokeStyle='#003057';cx.lineWidth=3;cx.stroke();
cx.font='bold 9px Arial';cx.fillStyle='#003057';cx.textAlign='center';
cx.fillText(sn.t.substring(0,20),x1,y1+r1+14);
cx.fillText(tn.t.substring(0,20),x2,y2+r2+14);
}}}}}}
function distColor(d){{
var ratio=Math.min(d/maxDist,1);
var r=Math.round(106+(212-106)*ratio);
var g=Math.round(177+(119-177)*ratio);
var b=Math.round(135+(107-135)*ratio);
return 'rgb('+r+','+g+','+b+')'}}
function nnHtml(items,show){{
return items.slice(0,show).map(function(t,i){{
var pct=Math.min(t.d/maxDist*100,100);
var isActive=tgt===t.s?'tgt-act':'';
var dimmed=t.d>=10&&!isActive;
var rowStyle=dimmed?'opacity:0.4;':'';
var nameColor=dimmed?'color:#999':'';
var socColor=dimmed?'color:#ccc':'';
var barBg=dimmed?'#ddd':distColor(t.d);
var distCol=isActive?'#fff':(dimmed?'#bbb':distColor(t.d));
return '<div class="nni '+isActive+'" style="'+rowStyle+'" onclick="event.stopPropagation();st(\\''+t.s+'\\','+t.d+')">'+
'<span class="nn-rank">'+(i+1)+'</span>'+
'<div class="nn-info"><div class="nn-name" style="'+nameColor+'">'+t.t+'</div><div class="nn-soc" style="'+socColor+'">'+t.s+'</div></div>'+
'<div class="nn-bar-wrap"><div class="nn-bar-bg"><div class="nn-bar-fill" style="width:'+pct+'%;background:'+barBg+'"></div></div></div>'+
'<span class="nn-dist" style="color:'+distCol+'">'+t.d+'</span></div>'}}).join('')}}
var showCounts={{}};
function rl(f){{
f=(f||'').toLowerCase();
var fl=N.filter(function(n){{return isVisible(n)&&(n.t.toLowerCase().indexOf(f)!==-1||n.s.indexOf(f)!==-1||n.c.toLowerCase().indexOf(f)!==-1)}});
document.getElementById('rc').textContent=fl.length+(f?' matching':' occupations');
document.getElementById('ol').innerHTML=fl.map(function(n){{
var ac=n.s===sel?'act':'',sh=n.s===sel?'show':'';
var sc=showCounts[n.s]||5;
var total=n.nn.length;
var moreBtn='';
if(n.s===sel && sc<total){{
if(sc<=5) moreBtn='<div class="nn-more" onclick="event.stopPropagation();showMore(\\''+n.s+'\\',15)">Show 10 more ▾</div>';
else if(sc<=15) moreBtn='<div class="nn-more" onclick="event.stopPropagation();showMore(\\''+n.s+'\\','+total+')">Show all ('+total+') ▾</div>';
}}
if(n.s===sel && sc>5) {{
moreBtn+='<div class="nn-more" onclick="event.stopPropagation();showMore(\\''+n.s+'\\',5)" style="color:#999">Collapse ▴</div>';
}}
return '<div class="card '+ac+'" onclick="so(\\''+n.s+'\\')">'+
'<div class="ctt">'+n.t+'</div><div class="csoc">'+n.s+'</div>'+
'<span class="cbadge" style="background:'+n.cl+'">'+n.c+'</span>'+
'<div class="mets"><div class="met"><div class="mv">'+(n.cti>0?'+':'')+n.cti.toFixed(3)+'</div><div class="ml">CTI</div></div>'+
'<div class="met"><div class="mv">'+n.dc.toFixed(3)+'</div><div class="ml">Degree</div></div>'+
'<div class="met"><div class="mv">'+n.cn+'</div><div class="ml">Connections</div></div></div>'+
'<div class="nns '+sh+'"><div class="nnt">Cross-Training Targets</div>'+
nnHtml(n.nn,sc)+moreBtn+'</div></div>'}}).join('')}}
function showMore(soc,count){{showCounts[soc]=count;rl(document.getElementById('sb').value)}}
function skillTags(nd){{
if(!nd||!nd.td||!nd.td.length)return '';
var tags=nd.td.slice(0,6).map(function(d){{
var cls=d.tp==='A'?'ab':'sk';
var lbl=d.tp==='A'?'(AB)':'(SK)';
return '<span class="ibar-skill '+cls+'" title="z='+d.z+'">'+lbl+' '+d.n+'</span>'}}).join('');
return '<div style="font-size:9px;color:#54585A;font-weight:700;margin-bottom:2px">DISTINCTIVE PROFILE <span style="font-weight:400;color:#999">(<span style="color:#6B4C1E">(AB)</span>=Ability <span style="color:#1B4965">(SK)</span>=Skill)</span></div><div class="ibar-skills">'+tags+'</div>'}}
function st(s,d){{
tgt=tgt===s?null:s;
var sn=sel?N.find(function(n){{return n.s===sel}}):null;
var tn=tgt?N.find(function(n){{return n.s===tgt}}):null;
if(sn&&tn){{
var distCol=d<6?'#2E7D32':(d<10?'#B3A369':'#B03A2E');
var costCol=d<6?'#2E7D32':(d<10?'#B3A369':'#B03A2E');
var costLbl=d<6?'Low — weeks of OJT':(d<10?'Moderate — months of structured training':'High — formal education / apprenticeship');
var costIcon=d<6?'&#9679;':(d<10?'&#9679; &#9679;':'&#9679; &#9679; &#9679;');
document.getElementById('ib').innerHTML='<div class="ibar-left"><div style="font-size:15px;font-weight:700;color:#003057;margin-bottom:6px">'+sn.t+' &rarr; '+tn.t+'</div><div style="display:flex;gap:24px;align-items:baseline;flex-wrap:wrap"><div><span style="font-size:10px;color:#999;font-weight:600">SKILL DISTANCE</span><div style="font-size:22px;font-weight:800;color:'+distCol+'">'+d+'</div></div><div><span style="font-size:10px;color:#999;font-weight:600">RETRAINING COST</span><div style="font-size:14px;font-weight:700;color:'+costCol+'">'+costIcon+' '+costLbl+'</div></div></div></div><div>'+skillTags(tn)+'</div>';
}}else if(sn){{
document.getElementById('ib').innerHTML='<div class="ibar-left"><div style="font-size:14px;font-weight:700;color:#003057">'+sn.t+' ('+sel+')</div><div style="font-size:12px;color:#555">Click a target to compare distance</div></div><div>'+skillTags(sn)+'</div>';
}}
rl(document.getElementById('sb').value);dr()}}
function so(s){{
sel=sel===s?null:s;tgt=null;if(!(s in showCounts))showCounts[s]=5;
var nd=N.find(function(n){{return n.s===s}});
if(sel&&nd){{
document.getElementById('ib').innerHTML='<div class="ibar-left"><div style="font-size:14px;font-weight:700;color:#003057">'+nd.t+' ('+s+')</div><div style="font-size:12px;color:#555">CTI: '+nd.cti+' | '+nd.cn+' connections | Click a target below</div></div><div>'+skillTags(nd)+'</div>';
}}else{{
document.getElementById('ib').innerHTML='<div class="ibar-left">Search or click an occupation to explore cross-training paths</div>';
}}
rl(document.getElementById('sb').value);dr()}}
document.getElementById('sb').addEventListener('input',function(e){{sel=null;tgt=null;rl(e.target.value);dr()}});
window.addEventListener('resize',rz);initCatCounts();rl();rz();
</script>
</body></html>"""

    path=os.path.join(OUT,'skill_explorer.html')
    with open(path,'w',encoding='utf-8') as f: f.write(html)
    fsize=os.path.getsize(path)
    print(f"  Saved: {path} ({fsize//1024}KB, {n} occupations)")
    print(f"  Fully self-contained — no internet needed, sharable as single file")

    td=sorted(deg.items(),key=lambda x:x[1],reverse=True)[:5]
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for s,v in td: print(f"    {titles.get(s,s)[:35]:<35s} DC={v:.3f}")

# =============================================================================
# MAIN
# =============================================================================
if __name__=='__main__':
    print("="*65)
    print("  SKILL DISTANCE CROSS-TRAINING FRAMEWORK")
    print("  Extends Project 1: Energy Belt Workforce Aging")
    print("="*65)
    df,titles=load_onet()
    gf,gc,cti,dz,dist,w,dims=compute(df)
    fig1_cti(df,gf,gc,cti,titles)
    build_html(df,dist,gf,gc,cti,titles)
    print(f"\n{'='*65}")
    print(f"  DONE — {len(df)} occupations")
    print(f"  {OUT}/fig1_cti_map.png    — Cognitive profile (50 SOCs)")
    print(f"  {OUT}/skill_explorer.html — Interactive tool (all SOCs)")
    print(f"{'='*65}")
