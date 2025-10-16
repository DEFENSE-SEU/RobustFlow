#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh


# =================================================================DROP
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=DROP --task=optimize --epoch=0
python evaluate.py --dataset=DROP --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=DROP --task=optimize --epoch=1
python evaluate.py --dataset=DROP --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py


sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/drop_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py

python generate.py --dataset=DROP --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/drop-dataset-1-test-heavy-noise.pkl

conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate

# =================================================================HumanEval
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=HumanEval --task=optimize --epoch=0
python evaluate.py --dataset=HumanEval --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=HumanEval --task=optimize --epoch=1
python evaluate.py --dataset=HumanEval --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py

# original
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

# requirements
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

# paraphrasing
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

# light-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

# moderate-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

# heavy-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/humaneval_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py
python generate.py --dataset=HumanEval --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/humaneval-dataset-1-test-heavy-noise.pkl

conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate


# =================================================================GSM8K
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=GSM8K --task=optimize --epoch=0
python evaluate.py --dataset=GSM8K --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=GSM8K --task=optimize --epoch=1
python evaluate.py --dataset=GSM8K --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py

# original
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

# requirements
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

# paraphrasing
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

# light-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

# moderate-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

# heavy-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/gsm8k_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py
python generate.py --dataset=GSM8K --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/gsm8k-dataset-1-test-heavy-noise.pkl


conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate


# =================================================================MATH
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=MATH --task=optimize --epoch=0
python evaluate.py --dataset=MATH --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=MATH --task=optimize --epoch=1
python evaluate.py --dataset=MATH --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py

# original
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

# requirements
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

# paraphrasing
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

# light-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

# moderate-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

# heavy-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/math_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py
python generate.py --dataset=MATH --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/math-dataset-1-test-heavy-noise.pkl

conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate

# =================================================================HotpotQA
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=HotpotQA --task=optimize --epoch=0
python evaluate.py --dataset=HotpotQA --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=HotpotQA --task=optimize --epoch=1
python evaluate.py --dataset=HotpotQA --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py

# original
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

# requirements
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

# paraphrasing
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

# light-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

# moderate-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

# heavy-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/hotpotqa_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py
python generate.py --dataset=HotpotQA --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/hotpotqa-dataset-1-test-heavy-noise.pkl

conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate

# =================================================================MBPP
cd ../ScoreFlow/
conda activate scoreflow

sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/" + bench_dic[data_set]["benchmark_name"] + "_" + data_set_type + ".jsonl"|' generate.py
sed -i '72s|.*|        post = "-test"|' evaluate.py
sed -i '121s|.*|    # with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        # f.write("original")|' evaluate.py
sed -i '126s|.*|        # f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test.txt"|' get_scores.py

python generate.py --dataset=MBPP --task=optimize --epoch=0
python evaluate.py --dataset=MBPP --task=optimize --epoch=0
accelerate launch --num_processes=1 optimize.py --epoch=0
python generate.py --dataset=MBPP --task=optimize --epoch=1
python evaluate.py --dataset=MBPP --task=optimize --epoch=1
accelerate launch --num_processes=1 optimize.py --epoch=1

sed -i '121s|.*|    with open("scoreflow_score.txt", "a") as f:|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '126s|.*|        f.write(f"Average Solve Rate: {sum([sub_l[2] for sub_l in avg_score_list])/len(avg_score_list)}\n")|' evaluate.py

# original
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_original.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-original"|' evaluate.py
sed -i '122s|.*|        f.write("original")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-original.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-original.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

# requirements
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_requirements.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-requirements"|' evaluate.py
sed -i '122s|.*|        f.write("requirements")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-requirements.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-requirements.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

# paraphrasing
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_paraphrasing.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-paraphrasing"|' evaluate.py
sed -i '122s|.*|        f.write("paraphrasing")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-paraphrasing.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-paraphrasing.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

# light-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_light_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-light-noise"|' evaluate.py
sed -i '122s|.*|        f.write("light-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-light-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-light-noise.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

# moderate-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_moderate_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-moderate-noise"|' evaluate.py
sed -i '122s|.*|        f.write("moderate-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-moderate-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-moderate-noise.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

# heavy-noise
sed -i '226s|.*|        output_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' generate.py
sed -i '231s|.*|    file_path = "data/mbpp_heavy_noise.jsonl"|' generate.py
sed -i '72s|.*|        post = "-test-heavy-noise"|' evaluate.py
sed -i '122s|.*|        f.write("heavy-noise")|' evaluate.py
sed -i '245s|.*|        input_dir = "scoreflow_workspace/output_workflow/dataset-" + str(epoch) + "-test-heavy-noise.pkl"|' get_scores.py
sed -i '246s|.*|        output_dir = "scoreflow_workspace/output_evaluation/scores-" + str(epoch) + "-" + str(parallel_id) + "-test-heavy-noise.txt"|' get_scores.py
python generate.py --dataset=MBPP --task=inference --epoch=1

cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-original.pkl        /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-original.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-requirements.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-requirements.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-paraphrasing.pkl    /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-paraphrasing.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-light-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-light-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-moderate-noise.pkl  /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-moderate-noise.pkl
cp /data/dishimin/RobustFlow/ScoreFlow/scoreflow_workspace/output_workflow/dataset-1-test-heavy-noise.pkl     /data/dishimin/RobustFlow/Evaluate/score_scripts/mbpp-dataset-1-test-heavy-noise.pkl

conda deactivate
cd ../Evaluate/
conda activate aflow
python eval_scoreflow.py
conda deactivate