#!/usr/bin/env bash
# Rebuild everything from the pinned seed, in dependency order.
#
# Verified end-to-end on 2026-09-05: regenerating produces byte-identical data and the
# same test-set SHA-256 (622d42b4fde8ed01...) recorded in reports/split_manifest.json.
#
# The committed repo already contains the data, the trained model and every report, so
# nothing here is required to run the service -- this exists so the numbers can be
# checked rather than taken on trust.
#
# Usage:  bash reproduce.sh
set -euo pipefail

PY="./.venv/Scripts/python.exe"          # on Linux/macOS: ./.venv/bin/python
export PYTHONPATH=src

step() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

step "1/8  Generate synthetic transactions (seed 7)"
$PY -m fraudspike.generate

step "2/8  Audit the data adversarially (no single marker may separate the classes)"
$PY -m fraudspike.audit_data

step "3/8  Assign the temporal split and pin the test-set hash"
$PY -m fraudspike.splits

step "4/8  Build 33 strictly backward-looking features"
$PY -m fraudspike.features

step "5/8  Audit features on the train split only"
$PY -m fraudspike.audit_features

step "6/8  Train and select on validation  (never reads the test split)"
$PY -m fraudspike.train

step "7/8  Score the held-out test set  (the only script that opens it)"
$PY -m fraudspike.evaluate

step "8/8  Cost analysis, decision bands, operating-point reconciliation"
$PY -m fraudspike.costs
$PY -m fraudspike.serving
$PY -m fraudspike.operating_point

step "Tests: causality and train/serve parity"
$PY -m pytest tests/ -q

printf '\n\033[1mDone.\033[0m Reports are in reports/. Start the service with:\n'
printf '  %s -m uvicorn api.app.main:app --port 8017\n' "$PY"
