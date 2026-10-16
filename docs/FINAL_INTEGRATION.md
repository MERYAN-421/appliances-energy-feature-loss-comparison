# Regression explorer integration

Baseline: 39585a0f5485b256ac2d8b7c415ec32f8843577b.
Preserve src's original 32-model engine and all outputs. Add a separate explorer,
training-only selection, validation K sweeps, OLS interpretation, saved-model live
prediction, a second problem, and a bounded custom CSV workspace.

The research pages remain available. New models never inherit the original
model's conformal calibration. Test data never choose selectors or K.
Milestones: interpretation; K sweep; Startups; custom data; integration tests.
