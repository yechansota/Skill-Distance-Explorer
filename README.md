# Skill Distance Cross-Training Framework for U.S. Manufacturing
<p align="center">
  <strong>An interactive tool for data-driven cross-training decisions in Energy Belt manufacturing</strong><br>
  <em>(Region: AL · GA · NC · SC · TN)</em><br>
</p>

[Try the live tool here](https://yechansota.github.io/Skill-Distance-Explorer/)

Following the project 1 analysis of workforce aging, identifying the problem is only the first step. This second project addresses the practical solution: what actions should we take? How can we practically resolve the aging workforce crisis in the manufacturing sector? Manufacturing is the backbone of the economy, accounting for roughly 30% of South Korea's GDP, and is deeply intertwined with the U.S. industrial characteristic. For the long-term sustainability of these operations, it is critical to implement highly efficient, cross-functional training driven by rational decision-making. In many manufacturing plants, vital decisions regarding staff training and job rotations often rely on the personal intuition of senior managers. While experience is valuable, it lacks the consistency needed for large-scale planning.

To solve this, I developed a data-driven framework that measures "skill gaps" between roles. This tool provides objective evidence for flexible staff transfers, targeted learning and development, and succession planning. The core idea is simple. While younger workers may learn new technologies faster, experienced employees possess irreplaceable institutional knowledge and judgment. This project isn't about replacement; it's about strategic placement. By moving veterans into roles where their wisdom is most effective, companies can maintain a competitive edge even as physical demands or technical tools evolve.


## The Theoretical Foundation: CHC Theory

The distance calculations in this tool are grounded in the **Cattell-Horn-Carroll (CHC) model** of human intelligence, the most widely validated framework in psychometric research ([Schneider & McGrew, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC9959556/)). CHC theory distinguishes between two broad categories of cognitive ability:

* **Fluid Intelligence (Gf):** The ability to reason with novel information, recognize patterns, and adapt to new situations. In manufacturing, this is vital for troubleshooting unfamiliar failure modes or learning new CNC programs. **Gf typically peaks in early adulthood and gradually declines with age.**
* **Crystallized Intelligence (Gc):** Accumulated knowledge, procedural expertise, and judgment built through years of experience. This allows a veteran mechanic to diagnose a failure just by sound. **Gc is preserved—and often grows—throughout a worker's career.**

### The Cognitive Tilt Index (CTI)

The **Cognitive Tilt Index (CTI)** is a metric I utilized for this project that captures where each occupation falls on the Gf-Gc spectrum. It is calculated as:

```
CTI = (Gc_mean - Gf_mean) / (Gc_mean + Gf_mean)
```

A **+** CTI means the job relies more heavily on crystallized **intelligence — experience, institutional knowledge, accumulated judgment**. These are the roles where losing a senior worker is most damaging, because what they carry in their heads cannot be easily taught in a training program. A **-** CTI means the job relies more on fluid **intelligence — adaptability, quick learning, pattern recognition**. These are roles where younger workers may actually ramp up faster, but where the aging workforce faces the steepest performance risks.

This distinction matters for workforce planning. When you know that a role has a CTI of +0.30 (heavily experience-dependent), you plan succession years in advance. When a role has a CTI of -0.15 (adaptability-driven), you can cross-train a capable worker in weeks rather than months.

### How CTI Works: The Normalized Difference Index

CTI is a novel metric — no prior study has applied this specific formulation to occupational data. However, the mathematical structure `(A - B) / (A + B)` is a **normalized difference index** that is one of the most widely validated functional forms in quantitative science such as NDVI and NPS.

This structure has 3 mathematical properties that make it well-suited for the CTI use case. First, the output is **bounded between -1 and +1**, which makes every occupation directly comparable regardless of its absolute score levels. Second, it is **scale-invariant** — a high-demand job (Gc=25, Gf=20) and a low-demand job (Gc=12.5, Gf=10) both yield the same CTI of +0.11, correctly reflecting that both lean toward Gc in the same proportion. Third, it is **symmetric around zero**, providing a natural decision boundary: positive = experience-dependent, negative = adaptability-dependent, zero = balanced.

The novelty of CTI is not the formula itself, but its application: **bridging CHC psychometric theory and O\*NET occupational data to quantify cognitive aging vulnerability at the job level**. CHC theory is established; the occupational mapping is the new contribution.


## Building the 87-Dimensional Occupational Vector

### Data Source: O\*NET Importance × Level

Each occupation in O\*NET is rated on 87 dimensions: **52 Abilities** (cognitive, physical, sensory, and psychomotor traits) and **35 Skills** (developed competencies). For every dimension, O\*NET provides two independent ratings from occupational analysts:

| Scale | What It Measures | Range | Example |
|---|---|---|---|
| **IM (Importance)** | How important is this ability/skill to the job? | 1 (not important) – 5 (extremely important) | Manual Dexterity for a Machinist: IM = 4.2 |
| **LV (Level)** | What level of this ability/skill is required? | 0 (not required) – 7 (highest level) | Manual Dexterity for a Machinist: LV = 5.1 |

Using either scale alone would lose information. A dimension can be "important but low-level" (IM=4, LV=2) or "moderately important but high-level" (IM=3, LV=5) — these are meaningfully different profiles. To capture both aspects in a single value, I compute a **composite score**:

```
C = IM × LV
```

This multiplicative composite means that a dimension only scores high when it is **both important and required at a high level**. A dimension that is unimportant (IM=1) stays low regardless of level, and a dimension required at a low level (LV=1) stays low regardless of importance.

### From Scores to Vectors

For each occupation, 87 composite scores are arranged into a single vector. Here is a simplified example for **Machinists (51-4041)**:

```
Dimension                    IM    LV    C = IM × LV
──────────────────────────────────────────────────────
Manual Dexterity            4.2   5.1     21.42
Visualization               3.8   4.6     17.48
Control Precision           4.0   4.3     17.20
Oral Comprehension          3.1   3.2      9.92
Written Comprehension       2.8   3.0      8.40
...
Programming                 1.5   1.2      1.80
──────────────────────────────────────────────────────
→ Vector: [21.42, 17.48, 17.20, 9.92, 8.40, ..., 1.80]
              (87 values total)
```

When this is done for all 150 occupations in the dataset, the result is a **150 × 87 matrix** — 150 points in 87-dimensional space. Occupations that require similar ability and skill profiles will be close together in this space; occupations with fundamentally different profiles will be far apart.

### Why O\*NET Uses Both Scales

O\*NET's dual-scale design reflects the reality that importance and level are independent constructs. A forklift operator needs Near Vision at a high importance (safety-critical) but moderate level (no fine detail work). A watchmaker needs Near Vision at both high importance and high level. The composite score correctly distinguishes these cases: the forklift operator might score IM=4 × LV=3 = 12, while the watchmaker scores IM=5 × LV=6 = 30.


## Mapping CHC Theory to O\*NET Abilities

Since O\*NET's 52 Abilities are not pre-labeled, I classified them based on CHC research literature ([McGrew, 2009](http://www.iapsych.com/chcdefs.pdf)).

The mapping follows a core principle:
> **If the ability requires on-the-spot reasoning with novel stimuli, it is Gf; if it draws on accumulated knowledge and practiced procedures, it is Gc.** This classification allows us to quantify the "cognitive tilt" of an occupation and predict how its performance requirements might interact with an aging workforce.

### Gf (Fluid Intelligence) — 11 Abilities

These are the abilities that peak in early adulthood and gradually decline with age. They represent a worker's capacity to handle the unexpected — new problems, unfamiliar patterns, situations where experience alone is not enough.

| O\*NET Ability | CHC Rationale | Manufacturing Example |
|---|---|---|
| **Inductive Reasoning** | Inferring general rules from specific patterns — core Gf | A maintenance tech notices three different machines failing on humid days and infers a moisture-related root cause |
| **Deductive Reasoning** | Applying general rules to reach specific conclusions — logical reasoning | An engineer determines that if the cooling system is rated for 200°F and the process runs at 220°F, the system will fail |
| **Category Flexibility** | Re-classifying information using different criteria — cognitive flexibility | A quality inspector switches from sorting defects by type to sorting by production line to identify the source |
| **Flexibility of Closure** | Identifying a known pattern embedded in distracting material — perceptual filtering | A CNC operator spots a subtle tool chatter pattern in a noisy vibration readout |
| **Speed of Closure** | Quickly making sense of incomplete or ambiguous information — rapid pattern completion | A power plant operator sees partial instrument readings during an alarm and quickly identifies the system state |
| **Perceptual Speed** | Rapidly comparing and identifying visual stimuli — processing speed | A packaging line operator spots a misaligned label on a bottle moving at 200 units/minute |
| **Spatial Orientation** | Maintaining awareness of one's position in space — spatial reasoning | A crane operator judges load swing trajectory relative to ground workers and obstacles |
| **Visualization** | Mentally rotating and manipulating 3D objects — mental manipulation | A machinist reads a 2D blueprint and visualizes how the finished part will fit into the assembly |
| **Memorization** | Encoding and retaining new information in working memory | A new hire memorizing a 15-step lockout/tagout procedure during their first week |
| **Fluency of Ideas** | Generating many ideas quickly in response to a prompt — divergent thinking | A process engineer brainstorms six different ways to reduce scrap rate on a new product line |
| **Originality** | Producing novel and useful solutions — creative reasoning | A maintenance mechanic fabricates a custom jig from scrap materials to hold an awkwardly shaped part during repair |

### Gc (Crystallized Intelligence) — 6 Abilities

These are the abilities that are preserved — and often continue to grow — throughout a worker's career. They represent the accumulated value of experience: knowing what to say, what to look for, and what order to do things in.

| O\*NET Ability | CHC Rationale | Manufacturing Example |
|---|---|---|
| **Oral Comprehension** | Understanding spoken information — draws on accumulated verbal and domain knowledge | A supervisor listens to an operator's description of an intermittent machine noise and immediately knows which subsystem to investigate |
| **Written Comprehension** | Understanding written information — reading-based knowledge access | A chemical technician reads a new SDS (Safety Data Sheet) and correctly interprets exposure limits based on years of working with similar compounds |
| **Oral Expression** | Communicating information effectively through speech — built on years of practice | A shift lead gives a clear, concise handoff briefing that covers exactly what the next shift needs to know |
| **Written Expression** | Communicating information effectively in writing — written expertise | A safety engineer writes an incident report that is both technically precise and understandable to non-engineers |
| **Problem Sensitivity** | Recognizing that something is wrong or is about to go wrong — experience-based intuition | A 25-year veteran hears a "different" sound from a motor and calls for shutdown 10 minutes before the bearing seizes |
| **Information Ordering** | Arranging information or actions in a logical sequence — procedural expertise | A production planner sequences 40 work orders to minimize changeover time, drawing on knowledge of which product transitions require full cleaning |

### Remaining 35 Abilities (Not in Gf/Gc)

The other abilities fall into Physical, Sensory, and Psychomotor groups. They are not part of the CTI calculation but are fully included in the 87-dimensional distance calculation with their own weights:

| Group | Count | Examples | Distance Weight |
|---|---|---|---|
| **Physical** | 9 | Static Strength, Stamina, Trunk Strength, Explosive Strength | 1.2x — physical demands are hard barriers to transition |
| **Sensory** | 12 | Near Vision, Depth Perception, Hearing Sensitivity, Color Discrimination | 1.1x — safety-critical, difficult to compensate |
| **Psychomotor** | 10 | Manual Dexterity, Finger Dexterity, Reaction Time, Rate Control | 1.0x — baseline |
| **Other Cognitive** | 4 | Selective Attention, Time Sharing, Number Facility, Mathematical Reasoning | 1.0x — baseline |

### 35 Skills

In addition to the 52 Abilities, O\*NET rates 35 Skills — developed competencies that fall between innate abilities and teachable knowledge. Examples include Operations Monitoring, Equipment Maintenance, Quality Control Analysis, Troubleshooting, Critical Thinking, and Complex Problem Solving. All 35 are included in the distance vector with CHC-aligned weights (see 2D weighting system below).


## Weighted Distance Calculation

### Step 1: Standardization (Z-score)

Raw composite scores are not directly comparable across dimensions because they have different natural ranges. "Static Strength" might average 15.2 with a standard deviation of 6.1 across all 150 occupations, while "Written Expression" might average 8.7 with a standard deviation of 4.3. A raw difference of 5.0 in Static Strength means something very different from a raw difference of 5.0 in Written Expression.

To solve this, every dimension is **z-score standardized** across all 150 occupations:

```
z = (raw_score - mean) / standard_deviation
```

After standardization, every dimension has mean = 0 and standard deviation = 1. A z-score of +1.5 means "1.5 standard deviations above the manufacturing average." This makes differences comparable across dimensions: a 1.0 difference in any z-scored dimension represents the same relative gap.

### Step 2: 2D Weighted Euclidean Distance

Not all dimensions are equally costly to bridge through training. Each dimension carries a weight determined by two factors: its **CHC cognitive alignment** (Gf, Gc, Physical, Sensory, or neutral) and its **trainability tier** (Ability vs. Skill). Abilities are "enduring attributes" that are harder to develop; Skills are "developed capacities" that are more trainable. The same cognitive direction therefore receives a lower weight when it appears as a Skill:

|  | Gf-aligned | Neutral | Gc-aligned | Physical-aligned |
|---|---|---|---|---|
| **Ability** (low trainability) | **1.5x** | 1.0x | **0.8x** | 1.2x (Physical) / 1.1x (Sensory) |
| **Skill** (moderate trainability) | **1.2x** | 1.0x | **0.9x** | **1.1x** |

Specific assignments:

| Weight | Dimensions | Count |
|---|---|---|
| **1.5x** | Gf Abilities: Inductive Reasoning, Deductive Reasoning, Category Flexibility, Flexibility of Closure, Speed of Closure, Perceptual Speed, Spatial Orientation, Visualization, Memorization, Fluency of Ideas, Originality | 11 |
| **1.2x** | Gf-like Skills: Critical Thinking, Complex Problem Solving, Active Learning, Learning Strategies | 4 |
| **1.2x** | Physical Abilities: Static Strength, Explosive Strength, Dynamic Strength, Trunk Strength, Stamina, Extent Flexibility, Dynamic Flexibility, Gross Body Coordination, Gross Body Equilibrium | 9 |
| **1.1x** | Sensory Abilities: Near Vision, Far Vision, Visual Color Discrimination, Night Vision, Peripheral Vision, Depth Perception, Glare Sensitivity, Hearing Sensitivity, Auditory Attention, Sound Localization, Speech Recognition, Speech Clarity | 12 |
| **1.1x** | Physical-aligned Skills: Equipment Maintenance, Repairing, Installation | 3 |
| **1.0x** | All other Abilities and Skills (Psychomotor, Other Cognitive, neutral Skills) | 41 |
| **0.9x** | Gc-like Skills: Reading Comprehension, Speaking, Writing, Active Listening, Instructing, Social Perceptiveness, Negotiation, Persuasion, Service Orientation | 9 |
| **0.8x** | Gc Abilities: Oral Comprehension, Written Comprehension, Oral Expression, Written Expression, Problem Sensitivity, Information Ordering | 6 |

The weighted Euclidean distance between occupation A and occupation B is:

```
distance(A, B) = √( Σᵢ wᵢ × (Aᵢ - Bᵢ)² )

where:
  i    = each of the 87 dimensions
  wᵢ   = weight for dimension i (from 2D table above)
  Aᵢ   = z-score of occupation A on dimension i
  Bᵢ   = z-score of occupation B on dimension i
```

### Step 3: Why the Weights Matter — A Worked Example

Consider the transition from Machinist (51-4041) to Welder (51-4121). Here is a simplified view of how individual dimensions contribute to the total distance:

```
Dimension              Group    Machinist(z)  Welder(z)  Diff    Weight  Contribution
─────────────────────────────────────────────────────────────────────────────────────
Inductive Reasoning    Gf         +0.8         +0.3      0.5     1.5     0.375
Visualization          Gf         +0.96        +0.2      0.76    1.5     0.867
Static Strength        Phys       -0.5         +1.8      2.3     1.2     6.348
Depth Perception       Sens       +0.3         +0.65     0.35    1.1     0.135
Oral Comprehension     Gc         +0.1         -0.2      0.3     0.8     0.072
Manual Dexterity       Psych      +1.25        +0.9      0.35    1.0     0.123
... (81 more dimensions)
─────────────────────────────────────────────────────────────────────────────────────
                                                    Total distance = √(sum of all)
```

Notice the asymmetry:
- The **Oral Comprehension gap** (Gc, 0.3 difference) contributes only **0.072** to the total because Gc gaps are cheap to close — a few months of on-the-job communication practice.
- The **Inductive Reasoning gap** (Gf, 0.5 difference) contributes **0.375** — nearly 5x more — because Gf gaps represent fundamental cognitive capacity differences that training cannot easily bridge.
- The **Static Strength gap** (Physical, 2.3 difference) contributes **6.348** — the largest single contributor — because welding demands significantly more upper-body strength, a hard physical barrier.

This is the core design principle: **the same numerical gap is treated differently depending on how trainable the underlying dimension is.** A 0.5-point gap in a Gf ability is a bigger obstacle than a 0.5-point gap in a Gc ability, and the distance metric reflects this.

### Why Knowledge (KN) Is Excluded from the Distance Calculation

| Component | Trainability | Development Method |
|---|---|---|
| **Knowledge** | Highest | Classroom instruction, OJT, self-study |
| **Skills** | Moderate | Repeated practice, structured training programs |
| **Abilities** | Lowest | Largely innate cognitive/physical traits — slow or impossible to change |

Including Knowledge in the distance calculation would **overestimate the true cost of cross-training transitions**. Consider the Machinist → Welder path: their knowledge domains differ substantially (metalworking theory vs. welding metallurgy), but their underlying ability profiles — manual dexterity, spatial visualization, arm-hand steadiness, rate control — are highly similar. In practice, a skilled machinist can learn welding knowledge through a structured OJT program in weeks, precisely because the ability foundation is already there.

The design principle is: **the distance should measure how hard it is to bridge the gap, not how different the domains look on paper.** Knowledge gaps are the easiest to close; ability gaps are the hardest.

Knowledge dimensions remain visible in the tool for a different purpose: they help managers understand *what domain-specific training content* a cross-training program should include, even when the distance score says the transition is feasible.


## How to Read the Distance Numbers

| Distance | What It Means | Practical Implication |
|---|---|---|
| Below 6 | Highly feasible | A structured OJT program of a few weeks is likely sufficient |
| 6 to 10 | Moderate | Requires a formal cross-training program over several months |
| Above 10 | Significant gap | Would likely require formal education, apprenticeship, or certification — treat as a new hire, not a transfer |

In the tool, distances above 10 are visually grayed out to help focus attention on actionable transitions.


## Practical Applications

This tool was built with four specific use cases in mind, all of which I encountered in my own HR work:

**Internal Transfers and Lateral Moves.** When an employee wants to move to a different role, or when a position opens up and you need to identify internal candidates, the tool shows which current roles have the shortest skill distance to the target. Instead of guessing, you can show the hiring manager that a Welding Machine Operator (51-4122) is only 4.8 distance units from a Machinist (51-4041), while a Material Handler (53-7062) is 11.3 units away. The data changes the conversation.

**L&D and Cross-Training Program Design.** When designing a cross-training curriculum, the tool helps prioritize which transitions to invest in. A plant that needs backup CNC operators should focus its training budget on machinists and tool grinders (distance under 5), not on assemblers (distance over 9).

**Succession Planning.** When a senior Tool & Die Maker (51-4111) with 28 years of experience is two years from retirement, the tool identifies which other skilled trades workers could most feasibly absorb their knowledge. The CTI score tells you how experience-dependent the role is, and the distance score tells you who is closest.

**Aging Workforce Retention.** For experienced workers whose physical capabilities are declining but whose knowledge is invaluable, the tool identifies lower-physical-demand roles within short cognitive distance. A veteran welder with a shoulder injury might transition to a Quality Inspector (51-9061) at a distance of 6.2, preserving their metallurgical knowledge in a role that requires less physical strain.


## Setup

```
pip install pandas numpy scipy matplotlib scikit-learn openpyxl networkx
```

Download `Abilities.xlsx` and `Skills.xlsx` from [O\*NET Database](https://www.onetcenter.org/database.html) and place them in the same directory as the script.

```
python3 skill_distance_final.py
```

The script produces two files in the `output_figures/` directory: a static cognitive profile map (`fig1_cti_map.png`) and the interactive HTML tool (`skill_explorer.html`). The HTML file is self-contained with no external dependencies and can be deployed directly to GitHub Pages or opened locally in any browser.


## Limitations and What I Would Do Next

### 1. Gf/Gc Mapping Is an Approximation, Not a CHC Factor Score

The Gf and Gc groupings used in this project are **approximate proxies** constructed from O\*NET's 52 Ability items, not direct CHC factor scores derived from psychometric test batteries (e.g., WJ-IV, WAIS-IV). Of the 11 abilities assigned to Gf, five belong to **adjacent CHC broad abilities** rather than Gf proper:

| Ability | CHC Proper Classification | Why I Included It in Gf |
|---|---|---|
| Visualization | **Gv** (Visual Processing) | In manufacturing, mental rotation of parts/assemblies functions as novel problem-solving; age-decline trajectory parallels Gf |
| Spatial Orientation | **Gv** (Visual Processing) | Crane operation, machine layout navigation require on-the-spot spatial reasoning under novel conditions |
| Memorization | **Gwm** (Working Memory) | Encoding new procedures is a Gf-adjacent capacity that declines with age |
| Fluency of Ideas | **Gr** (Retrieval Fluency) | Rapid idea generation for troubleshooting draws on Gf-like processing speed |
| Originality | **Gr** (Retrieval Fluency) | Novel solution generation in maintenance contexts involves creative reasoning |

These were included because their **occupational aging trajectories** resemble Gf decline more than their CHC-proper broad ability, but this broadened definition should be noted. A strict CHC Gf set would contain only 6 abilities (Inductive Reasoning, Deductive Reasoning, Category Flexibility, Flexibility of Closure, Speed of Closure, Perceptual Speed).

Similarly, the 6 Gc abilities primarily capture **verbal-communicative facets** of crystallized intelligence. O\*NET does not include direct measures of general knowledge depth (K0 — a core Gc narrow ability in CHC), because O\*NET measures that construct separately in its Knowledge section. The CTI therefore reflects a language-and-communication tilt rather than a full Gc profile.

The 11-vs-6 asymmetry may also introduce a **statistical bias**: Gf\_mean (averaged over 11 items) will have lower variance than Gc\_mean (averaged over 6 items), potentially producing noisier CTI values in the Gc direction.

**Sensitivity analysis completed (`fig_sensitivity.png`):** CTI rankings were compared across three Gf/Gc definitions (Broad 11/6, Intermediate 8/6, Strict 6/4). Spearman rank correlations: Broad vs Intermediate ρ=0.949, Broad vs Strict ρ=0.888, Intermediate vs Strict ρ=0.962. The Gc-dependent top-10 showed 6/10 overlap between Broad and Strict definitions; the Gf-dependent top-10 showed 3/10 overlap. **Conclusion: Gc-side rankings are robust to classification choice; Gf-side rankings are sensitive to whether Visualization and Memorization are included.** The current broadened definition is defensible for manufacturing contexts but should be noted as an upper-bound Gf estimate.

### 2. Skills Weighting — Resolved with 2D Weighting System

An earlier version of this tool assigned a uniform 1.0x weight to all 35 Skills, creating an internal inconsistency: a gap in "Deductive Reasoning" (Ability, weighted 1.5x) was penalized nearly twice as much as a gap in "Critical Thinking" (Skill, weighted 1.0x), despite both measuring essentially the same Gf-aligned cognitive construct.

This has been resolved by introducing the **2D weighting system** described above, which crosses CHC cognitive alignment with trainability tier. The 16 Skills with non-default weights are:

| Weight | Skills | CHC Alignment |
|---|---|---|
| **1.2x** | Critical Thinking, Complex Problem Solving, Active Learning, Learning Strategies | Gf-like: novel reasoning, new information processing |
| **1.1x** | Equipment Maintenance, Repairing, Installation | Physical-aligned: hands-on equipment handling |
| **0.9x** | Reading Comprehension, Speaking, Writing, Active Listening, Instructing, Social Perceptiveness, Negotiation, Persuasion, Service Orientation | Gc-like: language and experience-based communication |

**Validation:** Comparing distances before and after this change across all 11,175 occupation pairs showed a Pearson correlation of **0.9999**, mean absolute change of **0.042**, and **98.5% overlap** in each occupation's top-5 nearest neighbors. The theoretical consistency improved without disrupting any practical recommendations.

### 3. Distance Measures Transfer Cost, Not Total Retraining Cost

The distance model measures cognitive and physical ability transfer costs, not total retraining costs. Two roles can be "close" in ability space but require different certifications, licenses, or domain-specific training that the distance number does not capture. A Machinist and an Electrician might have similar ability profiles but completely different licensing requirements (e.g., state journeyman electrician license). The tool should be used as a starting point for conversation, not as the final decision.

### 4. Weights Are Theory-Driven, Not Empirically Calibrated
The CHC-based weights (Gf Ability=1.5x, Gc Ability=0.8x, Physical=1.2x, Sensory=1.1x, Gf-like Skill=1.2x, Gc-like Skill=0.9x) are grounded in the cognitive aging literature but have not been empirically calibrated with actual cross-training outcome data from manufacturing plants. The specific multipliers (why 1.5 and not 1.3 or 1.7?) are informed judgments, not regression coefficients derived from observed data.

### 5. CTI — External Validation Results
CTI was tested against BLS Current Population Survey data (Table 11b, 2024) for 82 matched manufacturing occupations. The initial hypothesis was that Gc-dependent jobs (positive CTI) would retain older workers longer, producing a positive correlation between CTI and % workers aged 55+.

The results showed a **significant negative correlation** (Pearson r=-0.309, p=0.005), contrary to the hypothesis. CTI vs median age showed no significant relationship (r=-0.059, p=0.67).

**Interpretation:** Workforce age structure is driven primarily by **labor market entry pathways and occupational tenure patterns**, not by cognitive aging vulnerability alone. High-CTI roles in manufacturing (HR specialists, production planners, accountants) attract younger college-educated entrants with high turnover, while lower-CTI skilled trades (machinists, welders, millwrights) retain workers for decades through apprenticeship systems and limited lateral mobility.

This finding does not invalidate CTI — it clarifies its proper scope. **CTI measures cognitive aging risk at the individual level** (what cognitive capacities will decline as this specific worker ages in this role), **not aggregate age composition** (how old the current workforce happens to be). A machinist role may have many older workers because of long tenure, but its moderate CTI correctly indicates that those workers face gradual Gf decline — they stay not because the job is age-friendly, but because switching costs are high. Conversely, a production planner role may have younger workers despite high CTI, but when those workers do age, their accumulated scheduling knowledge (Gc) will be the hardest asset to replace.

## Repository

```
Skill-Distance-Explorer/
├── index.html                   ← Interactive tool (open in browser or deploy to GitHub Pages)
├── README.md                    ← This file
├── skill_distance_final.py      ← Main analysis code (generates CTI map + interactive tool)
├── cti_validation.py            ← CTI external validation + sensitivity analysis
└── examples/
    ├── company_position_template.csv
    └── README_examples.md
```

---

Yechan Kim — Georgia Institute of Technology, M.S. Analytics (2026)
