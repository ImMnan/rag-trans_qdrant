APP_NAME := orca
VERSION ?=
IMAGE := immnan/orca-rag

GOOS := linux
GOARCH := amd64
PLATFORM := $(GOOS)/$(GOARCH)

BINARY_DIR := bin
BINARY := $(BINARY_DIR)/$(APP_NAME)-$(GOOS)-$(GOARCH)
DOCKERFILE := container/Containerfile

.PHONY: help check-version build-binary build-image push-image release clean

help:
	@echo "Usage: make <target> VERSION=<tag>"
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
	cd rag-go && CGO_ENABLED=0 GOOS=$(GOOS) GOARCH=$(GOARCH) \
		go build -ldflags="-s -w -X main.orcaVersion=$(VERSION)" -o ../$(BINARY) ./main
	@echo "Built binary: $(BINARY)"

build-image: check-version
	docker buildx build \
		--platform $(PLATFORM) \
		--load \
		-f $(DOCKERFILE) \
		--build-arg VERSION=$(VERSION) \
		-t $(IMAGE):$(VERSION) \
		.

push-image: check-version
	docker push $(IMAGE):$(VERSION)

release: build-image push-image

clean:
	rm -rf $(BINARY_DIR)
