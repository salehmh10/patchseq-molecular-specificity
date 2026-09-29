# Methods overview

The analysis joins paired electrophysiology and transcriptomics through
canonical specimen identifiers. The fixed cohort comprises 3,410 mouse
visual-cortex inhibitory neurons from 871 donor groups. Released feature
standardization can be inverted only up to the upstream clipping boundaries.
All downstream imputation, scaling, nuisance fits, and model selection use
training rows only.

Six prespecified transcript modules are unweighted means of log2(CPM + 1).
Library denominators include all 45,768 count rows. Exact-symbol coverage is
95 of 96 requested genes. NaV6 uses six observed symbols; NaV7 adds the
release-native `Scn2a1` as a distinct sensitivity target. No inferred alias
silently replaces the original definition.

The original feature set has 24 predictors. The final 23-feature analysis
retains rheobase and removes its verified exact stimulus-amplitude duplicate.
Waveform principal components are excluded. Elastic Net, XGBoost, and an MLP
use identical three-fold donor-disjoint evaluation, with a donor-disjoint
inner selection split. Model specifications are recorded in `config/` and
Supplement Tables S7–S9.

Matched gene sets preserve target size and match mean expression and detection
rate. They do not directly match co-expression. The 1,000-control extension
uses plus-one upper-tail empirical p-values and BH over exactly six raw-target
tests. Technically residualized specificity forms a separate six-test family.
Control predictions use the target's selected XGBoost configuration; retuning
is a separate sensitivity analysis.

Paired model inference uses 5,000 donor-cluster bootstrap draws and 18 primary
R² contrasts. Saved-prediction permutations are association diagnostics without
model refits. Internal consistency, subclass dependence, gene-definition
sensitivity, and technical-target prediction qualify interpretation.

Results concern predictive associations. They do not identify unique molecular
mechanisms or establish protein levels, membrane localization, or conductance.
