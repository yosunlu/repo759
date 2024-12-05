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


cleanup:
	rm -f *.out
	rm -f *.err
	rm -f *.txt
	rm -f task1
	rm -f task2