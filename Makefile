PYTHON ?= python3
PORT ?= 8501
HOST ?= 0.0.0.0
CHECKPOINT_URL ?= https://minio1.webtui.vn:9000/browser/bucket-vmt/checkpoint%2Fwider_resnet38.pth
CHECKPOINT_DIR ?= checkpoints
FORCE ?= 0

.DEFAULT_GOAL := ui

.PHONY: ui checkpoint checkpoints download-checkpoint help

ui:
	@echo "==> Launching Video Anomaly Detection UI on $(HOST):$(PORT)..."
	streamlit run app.py --server.port=$(PORT) --server.address=$(HOST)

checkpoint:
	@$(PYTHON) scripts/download_checkpoint.py --url "$(CHECKPOINT_URL)" --dest "$(CHECKPOINT_DIR)" $(if $(filter 1 true TRUE yes YES,$(FORCE)),--force,)

checkpoints: checkpoint
download-checkpoint: checkpoint

help:
	@echo "Available make targets:"
	@echo "  make ui                    Launch Streamlit web application (PORT=$(PORT) HOST=$(HOST))"
	@echo "  make checkpoint            Download pretrained checkpoint (wider_resnet38.pth)"
	@echo "                             Options: FORCE=1 to force re-download"
	@echo "                                      CHECKPOINT_URL=<url> to override URL"
	@echo "                                      CHECKPOINT_DIR=<dir> to override destination"
	@echo "  make help                  Show this help message"
