@echo off
cd /d %~dp0\..

echo === Running robustness GMM experiments ===

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.1 jko_steps=40 seed=1 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_004 alpha=0.1 jko_steps=40 seed=1 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_008 alpha=0.1 jko_steps=40 seed=2 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_016 alpha=0.1 jko_steps=40 seed=2 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_032 alpha=0.1 jko_steps=40 seed=2 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_064 alpha=0.1 jko_steps=40 seed=2 crank_nicolson=False
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_128 alpha=0.1 jko_steps=40 seed=3 crank_nicolson=False

echo === All experiments completed and stored ===