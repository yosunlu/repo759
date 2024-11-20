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

task2:
	chmod +x ./scripts/task2.sh
	sbatch ./scripts/task2.sh

task2_c:
	chmod +x ./scripts/task2_c.sh
	sbatch ./scripts/task2_c.sh

cleanup:
	rm -f *.out
	rm -f *.err
	rm -f *.txt
	rm -f task1
	rm -f task2
	
