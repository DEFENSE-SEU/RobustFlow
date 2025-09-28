cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/mbpp_original.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/mbpp_requirements.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/mbpp_paraphrasing.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/mbpp_noise.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_original_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_original_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_original_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_original_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_original_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_original_5/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_requirements_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_requirements_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_requirements_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_requirements_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_requirements_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_requirements_5/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_paraphrasing_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_paraphrasing_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_paraphrasing_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_paraphrasing_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_paraphrasing_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_paraphrasing_5/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_noise_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_noise_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_noise_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_noise_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_noise_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_noise_5/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_light_noise_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_light_noise_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_light_noise_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_light_noise_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_light_noise_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_light_noise_5/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_moderate_noise_1.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_1/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_1/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_moderate_noise_2.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_2/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_2/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_moderate_noise_3.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_3/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_3/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_moderate_noise_4.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_4/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_4/

cd /data/dishimin/RobustFlow/AFlow/workspace/
rm -rf MBPP/
tar -xzf MBPP.tar.gz
cd ../data/datasets/
rm -rf mbpp_validate.jsonl
cp /data/dishimin/RobustFlow/Noise_Dataset/MBPP/SUB/mbpp_moderate_noise_5.jsonl /data/dishimin/RobustFlow/AFlow/data/datasets/mbpp_validate.jsonl
cd ../../
python run.py --dataset MBPP --check_convergence False
mkdir -p /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_5/
cp -a /data/dishimin/RobustFlow/AFlow/workspace/MBPP/workflows/. /data/dishimin/RobustFlow/Evaluate/aflow_scripts/MBPP/mbpp_moderate_noise_5/
