PORT ?= 8501
HOST ?= 0.0.0.0

.DEFAULT_GOAL := ui

.PHONY: ui

ui:
	@echo "==> Launching Video Anomaly Detection UI on $(HOST):$(PORT)..."
	streamlit run app.py --server.port=$(PORT) --server.address=$(HOST)
