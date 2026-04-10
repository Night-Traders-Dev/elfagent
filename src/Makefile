.PHONY: install prefetch run lint tree experiments

install:
	python3 -m pip install -r requirements.txt

prefetch:
	python3 install_models.py

run:
	python3 agent.py

lint:
	python3 -m py_compile agent.py core/*.py ui/*.py summary/*.py tools/*.py integrations/*.py utils/*.py routing/*.py helpers/*.py reasoning/*.py orchestration/*.py experiments/*.py

tree:
	find . -maxdepth 3 -type f | sort

experiments:
	python3 experiments/run_prompt_experiments.py

benchmark:
	python3 benchmarks/latency_benchmark.py

eval-routes:
	python3 evaluation/route_eval.py
