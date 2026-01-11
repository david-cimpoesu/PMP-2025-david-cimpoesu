# Partial exam grading notes

Checked the provided `Partial/solutions_partial.py` against `Partial/Partial.pdf`: the solution matches the final exam text (the score table in the PDF mislabels “2.c” twice; actual tasks are 1.a/1.b/1.c and 2.a/2.b/2.c/2.d + 1 ex officio).

## Rubric (max 30 + 1 ex officio)
- **1.a (4p)**: BN structure O→H, O→W, H→R, W→R, H→E, R→C with correct CPDs; partial credit for minor CPD slips.
- **1.b (5p)**: VE queries P(H=yes|C=comfortable), P(E=high|C=comfortable), MAP(H,W|C=comfortable); deduct for missing pieces or clearly wrong CPDs.
- **1.c (4p)**: d-sep reasoning for W ⟂ E | H and O ⟂ C | R. If explanations are missing in code, noted below to check paper.
- **2.a (4p)**: HMM defined with given π, A, B.
- **2.b (4p)**: Forward probability for [M, H, L] (±1e-3 accepted).
- **2.c (4p)**: Viterbi path for [M, H, L] + short rationale on Viterbi vs brute force; partial if rationale missing.
- **2.d (4p)**: ~10k simulations to estimate P([M, H, L]) and compare to forward.
- **Ex officio (1p)**: clarity/organization (plots/screens/markdown).

## Updated grades (re-evaluated generously, rows ≤ Braha Petru)
- **Bazon Bogdan** (`sky11fca/Bazon-Bogdan-PMP-2025`) → **17/30** — BN structure ok but CPDs flipped; VE queries done; no 1.c; HMM forward fine, Viterbi points for intent despite bugged call; no sim.
- **Bulai Vali-Danuț** (`danutbulai/PMP-2025`) → **22/30** — BN+queries correct; 1.c absent; HMM forward+Viterbi solid; no simulation.
- **Burcă Sabrina-Valentina** (`SabrinaV12/PMP`) → **12/30** — BN adds E→C and comfort CPD off; VE/MAP run; 1.c missing; HMM params off but forward computed; no Viterbi/sim.
- **Chirica Sebastian** (`SebyThePro/ChiricaSebastianPMP`) → **22/30** — BN+VE ok; 1.c missing; HMM forward+empirical sim present; Viterbi not done.
- **Cimpoesu David-Richard** (`david-cimpoesu/PMP-2025-david-cimpoesu`) → **21/30** — BN prior O off and only H|C query; no 1.c; HMM forward/Viterbi/sim all implemented.
- **Cires Vlad** (`Dvkan/PMP-2025`) → **17/30** — BN CPDs off; no 1.c; HMM forward+Viterbi attempted; no simulation on target sequence.
- **Cozma Damian-Constantin** (`damian-cozma/PMP-2025-Damian`) → **26/30** — Strong BN/VE and d-sep notes; HMM forward+Viterbi with rationale; sim not finished.
- **Crăciun Daniel** (`DannyDaniel-03/PMP-2025`) → **28/31** (incl. +1 ex officio) — Full BN+queries; 1.c not explicit but rest clean; HMM forward/Viterbi/sim complete and tidy.
- **Croitor Vladimir-Alexandru** (`BigALL-IN/PMP-Croitor-Vladimir-Alexandru-3E2`) → **20/30** — BN CPDs off; 1.c partly addressed (W⊥E|H only); HMM forward+Viterbi ok; no sim.
- **Iancu Ștefan-Teodor** (`iancustefan26/Probabilistic-Programming-and-Modeling-Lab`) → **23/30** — BN/VE fine; d-sep checks present; HMM forward and Viterbi via predict; no sim on [M,H,L].
- **Ioniche Adrian** (`ionicheadrian/PMP-2025`) → **17/30** — BN structure/CPDs off and no VE; HMM forward/Viterbi/sim completed.
- **Iordache George-Matei** (`ZzeLDor/PMP-2025`) → **20/30** — BN CPDs off; 1.c light; HMM forward+Viterbi computed; no sim.
- **Marțișcă Diana-Maria** (`dianamartisca/PMP-2025`) → **19/30** — BN+VE ok; 1.c minimal; HMM only Viterbi path (no forward/sim).
- **Minuț Alexandru-Tudor** (`TudorMinut/PMP-2025-Minut-Alexandru-Tudor`) → **18/30** — BN CPDs partly off; no 1.c; HMM swaps states/obs but runs forward/Viterbi/sim on that setup.
- **Mototolea Cosmin David** (`mototoleacosmindavid/PMP`) → **24/30** — BN CPD flipped for H; 1.c answered; HMM forward+Viterbi good; no sim.
- **Shahin Wissam** (`DragonLordKing/PMP-2025`) → **19/30** — BN CPDs wrong; 1.c missing; HMM uses longer sequence but forward/Viterbi/sim all implemented.
- **Stănulet Rudolf-Sebastian** (`VexWarlock/PMP-2025`) → **16/30** — Only HMM submitted; forward/Viterbi/sim correct.
- **Vezeteu Andrei** (`JustAnotherAndrei/PMP2025`) → **31/31** (incl. +1 ex officio) — Full, correct BN+VE+d-sep; HMM forward/Viterbi/sim all correct with clear write-up.
- **Bitiușcă (Manucă) Mădălina** (`madalinabiti/PMP-2025`) → **10/30** — BN/HMM params off but model + forward computed; 1.c and Viterbi/sim missing.
- **Ghența Ana Maria** (`AnaMariaGh1/PMP`) → **5/30** — Partial BN (no Comfort CPD) and queries would fail; HMM not set per spec; minimal credit for partial setup.
- **Braha Petru** (`petru-braha/class-PMP`) → **25/30** — BN+VE good; 1.c partly off; HMM forward+Viterbi ok; simulation not included.
