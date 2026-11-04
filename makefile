SERVICE ?= orca
VERSION ?=

ifeq ($(SERVICE),orca)
APP_NAME := orca
IMAGE := immnan/orca-rag
MODULE_DIR := rag-go
BUILD_PACKAGE := ./main
DOCKERFILE := container/Containerfile
else ifeq ($(SERVICE),feedback)
APP_NAME := orca-feedback
IMAGE := immnan/orca-feedback
MODULE_DIR := feedback
BUILD_PACKAGE := .
DOCKERFILE := container/Containerfile.feedback
else
$(error Unsupported SERVICE "$(SERVICE)"; use orca or feedback)
endif

GOOS := linux
GOARCH := amd64
PLATFORM := $(GOOS)/$(GOARCH)

BINARY_DIR := bin
BINARY := $(BINARY_DIR)/$(APP_NAME)-$(GOOS)-$(GOARCH)

.PHONY: help check-version build-binary build-image push-image release clean

help:
	@echo "Usage: make <target> VERSION=<tag> [SERVICE=orca|feedback]"
	@echo "SERVICE defaults to orca. Example: make release SERVICE=feedback VERSION=0.1.0"
	@echo
	@echo "Available targets:"
	@echo "  build-binary  - Build $(PLATFORM) Go binary with the supplied version"
	@echo "  build-image   - Build $(IMAGE):<version>"
	@echo "  push-image    - Push $(IMAGE):<version> to Docker Hub"
	@echo "  release       - Build and push $(IMAGE):<version>"
	@echo "  make clean         - Remove local build artifacts"

check-version:
	@test -n "$(VERSION)" || (echo "VERSION is required. Example: make release VERSION=1.2.3" && exit 1)

build-binary: check-version
	@mkdir -p $(BINARY_DIR)
	cd $(MODULE_DIR) && CGO_ENABLED=0 GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="-s -w -X main.orcaVersion=$(VERSION)" -o ../$(BINARY) $(BUILD_PACKAGE)
	@echo "Built binary: $(BINARY)"

build-image: check-version
	docker build \
		--platform $(PLATFORM) \
		-f $(DOCKERFILE) \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE):$(VERSION) \
		.

push-image: check-version
	docker push $(IMAGE):$(VERSION)

release: build-image
	$(MAKE) push-image SERVICE=$(SERVICE) VERSION=$(VERSION)

clean:
	rm -rf $(BINARY_DIR)
