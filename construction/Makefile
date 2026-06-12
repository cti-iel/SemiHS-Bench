PYTHON ?= python3

# The raw observations (data/raw/) and the per-version release/ working pools
# are not redistributed (see README.md), so the build targets below only run
# where those inputs exist. `make test` runs everywhere.

normalize-rulings:
	$(PYTHON) scripts/normalize_jp_customs.py

ingest-catalog:
	$(PYTHON) scripts/ingest_catalog.py

ingest-bol:
	$(PYTHON) scripts/ingest_bol.py

corpus:
	$(PYTHON) scripts/build_reference_corpus.py

worksheet:
	$(PYTHON) scripts/generate_review_worksheet.py

prefill-core4:
	$(PYTHON) scripts/prefill_core4_catalog.py

apply-audit:
	$(PYTHON) scripts/apply_audit_decisions.py

dossiers:
	$(PYTHON) scripts/build_hs6_dossiers.py

calibration:
	$(PYTHON) scripts/build_calibration_set.py

annotate-difficulty:
	$(PYTHON) scripts/annotate_difficulty.py

iaa-export:
	$(PYTHON) scripts/iaa_export_csv.py

part1:
	$(PYTHON) scripts/build_part1_release.py

eval-set:
	$(PYTHON) scripts/build_eval.py

release:
	$(PYTHON) scripts/build_release.py

test:
	$(PYTHON) -m pytest tests -q

.PHONY: normalize-rulings ingest-catalog ingest-bol corpus worksheet \
        prefill-core4 apply-audit dossiers calibration annotate-difficulty \
        iaa-export part1 eval-set release test
