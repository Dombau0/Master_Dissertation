@echo off
cd /d %~dp0\..

echo === Running 2-dimensional GMM experiments ===

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=2.0 jko_steps=2 seed=0,1,2,3,4 crank_nicolson=True
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=2.0 jko_steps=2 seed=0,1,2,3,4 crank_nicolson=False

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=1.0 jko_steps=4 seed=0,1,2,3,4 crank_nicolson=True
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=1.0 jko_steps=4 seed=0,1,2,3,4 crank_nicolson=False

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.5 jko_steps=8 seed=0,1,2,3,4 crank_nicolson=True
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.5 jko_steps=8 seed=0,1,2,3,4 crank_nicolson=False

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.2 jko_steps=20 seed=0,1,2,3,4 crank_nicolson=True
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.2 jko_steps=20 seed=0,1,2,3,4 crank_nicolson=False

python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.1 jko_steps=40 seed=0,1,2,3,4 crank_nicolson=True
python src_gmm\gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.1 jko_steps=40 seed=0,1,2,3,4 crank_nicolson=False

echo === All experiments completed and stored ===