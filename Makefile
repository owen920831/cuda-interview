.PHONY: setup build test test-tools check profile-clean docker-env docker-build docker-test docker-check docker-shell

setup:
	./scripts/check_env.sh

build:
	cmake --preset release
	cmake --build --preset release -j

test:
	ctest --preset release

test-tools:
	python3 -m unittest \
		tools/test_perf_calculator.py \
		tools/test_llm_calculator.py \
		tools/test_batching_simulator.py

check:
	python3 tools/check_repo.py
	python3 -m unittest \
		tools/test_perf_calculator.py \
		tools/test_llm_calculator.py \
		tools/test_batching_simulator.py

profile-clean:
	find out -type f \( -name '*.ncu-rep' -o -name '*.nsys-rep' -o -name '*.sqlite' \) -delete 2>/dev/null || true

docker-env:
	./scripts/docker.sh env

docker-build:
	./scripts/docker.sh build

docker-test:
	./scripts/docker.sh test

docker-check:
	./scripts/docker.sh check

docker-shell:
	./scripts/docker.sh shell
