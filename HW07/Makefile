push:
	git add .
	git commit -m "$(MSG)"
	git push

pull:
	make cleanup
	git reset --hard HEAD
	git pull
	
task1:
	chmod +x ./scripts/task1.sh
	sbatch ./scripts/task1.sh

task1_b:
	chmod +x ./scripts/task1_b.sh
	sbatch ./scripts/task1_b.sh

task2:
	chmod +x ./scripts/task2.sh
	sbatch ./scripts/task2.sh

task2_b:
	chmod +x ./scripts/task2_b.sh
	sbatch ./scripts/task2_b.sh

cleanup:
	rm -f *.out
	rm -f *.err
	rm -f *.txt
	rm -f task1
	rm -f task2