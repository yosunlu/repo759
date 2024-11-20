.PHONY: task1

push:
	git add .
	git commit -m "$(MSG)"
	git push

pull:
	git reset --hard HEAD
	git pull

task1:
	chmod +x ./scripts/task1.sh
	sbatch ./scripts/task1.sh

task1_c:
	chmod +x ./scripts/task1_c.sh
	sbatch ./scripts/task1_c.sh

output1:
	cat ./output/task1.err
	cat ./output/task1.out

task2:
	chmod +x ./scripts/task2.sh
	sbatch ./scripts/task2.sh

output2:
	cat ./output/task2.err
	cat ./output/task2.out
