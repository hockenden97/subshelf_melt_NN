#!/bin/bash

nemo_run='isfru94' # Options are 'OPM016', 'OPM018', 'OPM021', 'ctrl94', 'isf94', 'isfru94'
year_min=2014
year_max=2020
job_type='PREP_TSmelt'

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/PREP_ALL_TSmelt.py

# Where to save the job output
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/JOB_files/
path_local=JOB_files/

# Where to save the job output
path_jobname=$path_${job_type}_${nemo_run}_${year_min}_${year_max}
echo "Running these variables: " $path_jobname
echo "path" $path
path_sh_file=$path_jobid${job_type}_${nemo_run}_${year_min}_${year_max}
path_sh_local=$path_local${job_type}_${nemo_run}_${year_min}_${year_max}
echo "path_sh_file" $path_sh_file

# Define the job that will run (load environment, save python output to log file) 
cat <<EOF > $path_sh_file.sh
#!/bin/bash
# Load the conda environment
# conda init
conda activate nnets_py38
# Run python with the specified variables 
# The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
python -u $path_python ${nemo_run} ${year_min} ${year_max} 2>&1 
echo 'Finished' $OAR_JOB_ID 
echo 'Finished' $OAR_JOB_ID 1>&2
EOF

# Make the job file executable
chmod +x $path_sh_file.sh

# And then execute it 
oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=4,walltime=4:00:00 -n $path_jobname --project mais 

# And then remove the sh file which runs the code because they clutter up the folder and are all just repeats 
#rm $path_jobname.sh
