OUTPUT_PATH=./output
mkdir -p $OUTPUT_PATH

python openai_judge.py #&> $OUTPUT_PATH/openai_judge.log
