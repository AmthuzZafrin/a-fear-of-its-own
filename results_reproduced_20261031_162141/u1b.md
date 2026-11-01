# U1b: fear versus a belief-only twin

n: agents 10 (40-agent set 40), twins 10 (all taus 20)
mean lifetime: agents 11870, twins 9980

| test | agents mean [95% CI] | Holm p (P1) | twins mean [95% CI] | agents - twins | Holm p (P2) |
|---|---|---|---|---|---|
| T1_wounded (pred +) | +0.238 [+0.076, +0.401] | 0.009 | +0.022 [+0.007, +0.037] | +0.216 | 0.019 |
| T1_hungry (pred -) | -0.234 [-0.355, -0.114] | 0.0052 | -0.059 [-0.090, -0.029] | -0.175 | 0.019 |
| T2_wounded (pred +) | +6.178 [+2.715, +9.641] | 0.0059 | +0.092 [-0.180, +0.365] | +6.086 | 0.0096 |

40-agent set (secondary): T1_wounded +0.058 [-0.082, +0.198] p=0.41; T1_hungry -0.240 [-0.298, -0.183] p=2.7e-10; T2_wounded +6.272 [+2.332, +10.212] p=0.0026

P3_twin: a persistence +0.633 (p=1.9e-10); b duration slope +2.08 (p=3.9e-15; hl by tau {'4.0': 1.9387243105382876, '12.0': 3.8841288452675498, '36.0': 6.518750416751777}); c ablation log10 +1.63 (p=4.7e-12); d hunger +0.095 (p=5e-06); passed 4/4
T3_agents_reference: a persistence +0.575 (p=1.7e-09); b duration slope +3.03 (p=5e-10; hl by tau {'4.0': 2.5257418735694968, '12.0': 4.935700249957376, '36.0': 9.178768708147372}); c ablation log10 +1.62 (p=2.5e-13); d hunger +0.391 (p=3.5e-06); passed 4/4

twin memory: mean held-out AUC 0.917, readout R2 vs ideal log-odds 0.994

decision: {'stakes_tests_passed': ['T1_wounded', 'T1_hungry', 'T2_wounded'], 'lead': 'stakes law'}
