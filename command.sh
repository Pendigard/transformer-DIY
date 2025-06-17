/Vrac/renton/MachineLearning/bin/python main.py --save_name fr_en_default --lang1 fr --lang2 en --token_path ./tokenizer/tok_tatoeba_en_fr.json --data_path ./data/tatoeba_en_fr --batch_size 64 --max_epochs 20 --patience 3 --learning_rate 1e-4 --dropout 0.1 --d_model 512 --d_ff 2048 --h 8

python3 translate.py --save_name fr_en_default --sentence "