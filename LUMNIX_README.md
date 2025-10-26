# create & activate venv (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate

# install deps and the package
pip install -r requirements.txt
python3 -m pip install -e .

# run a short synthetic simulation (128 requests, 60s time limit)
python -m vidur.main \
  --synthetic_request_generator_config_num_requests 1 \  
  --time_limit 60 \
  --metrics_config_output_dir simulator_output/test_run