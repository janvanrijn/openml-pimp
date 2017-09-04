export PYTHONPATH=/home/vanrijn/projects/openml-python2:/home/vanrijn/projects/Study-14/Python:/home/vanrijn/projects/openml-pimp
#!/bin/bash
while true
do
	/home/vanrijn/projects/pythonvirtual/pimp/bin/python3.5 /home/vanrijn/projects/openml-pimp/openmlpimp/generatedata/generate.py --openml_apikey d488d8afd93b32331cf6ea9d7003d4c3 --study_id 14 --classifier libsvm_svc
done
