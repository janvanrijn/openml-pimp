export PYTHONPATH=/home/vanrijn/projects/openml-python2:/home/vanrijn/projects/Study-14/Python:/home/vanrijn/projects/openml-pimp
while :
do
  /home/vanrijn/projects/pythonvirtual/pimp/bin/python3.5 /home/vanrijn/projects/openml-pimp/openmlpimp/generatedata/run_default.py --openml_apikey d488d8afd93b32331cf6ea9d7003d4c3 --classifier libsvm_svc --fixed_parameters '{"kernel": "sigmoid"}'
done
