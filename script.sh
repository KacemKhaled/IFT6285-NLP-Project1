#!/bin/bash
# albert-base-v2 distilbert-base-cased gpt2 bert-base-uncased bert-base-cased distilbert-base-uncased-finetuned-sst-2-english roberta-large  roberta-large
#bert-base-uncased distilbert-base-cased distilbert-base-uncased

# albert-base-v2 bert-base-cased roberta-large bert-large-cased 

# task = MRPC
t=mrpc 
for a in albert-base-v2 bert-base-cased roberta-base albert-large-v2 bert-large-cased roberta-large ; do
	for lr in  6e-5  5e-5  4e-5 2e-5 ; do
		for b in 64 32 16 8 ; do
			e=1
			logfile=task-$t--arch-$a--epochs-$e
			echo "-----------------------------------------------------------------------"
			echo "Training --arch $a --task $t --epochs $e "
			python test-transformers.py --arch $a --task $t --epochs $e --batch_size $b --lr $lr #2> logs/$log_file.log
			echo "-----------------------------------------------------------------------"
		done
	done
done

# #python test-transformers.py --arch albert-base-v2 --task cola --epochs 2


#python test-transformers.py --arch albert-base-v2 --task cola --epochs 2
# best mdoels 