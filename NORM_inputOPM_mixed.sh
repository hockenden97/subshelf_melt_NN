#!/bin/bash
#this_collection_NN='OPM026_to2028_OPM031_2089to2098' #'OPM026_whole_dataset'
#this_collection_NN='OPM026_OPM031_1yr'
#this_collection_NN='OPM026_OPM031_5yr'

echo "Merging multiple simulations"
this_collection='OPM026_OPM0263_OPM031_OPM016_OPM018_OPM021_ctrl94_isf94_isfru94' #_OPM031_Christoph"
job_type='NORM_'
echo ${this_collection}

# Where to find the python script to run the job on 
path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/NN_norm_input_OPM_mixed_fullruns.py
path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/JOB_files/
path_local=JOB_files/

# Where to save the job output
path_jobname=$path_${job_type}${this_collection}
echo "Running these variables"

path_sh_file=${path_jobid}${job_type}${this_collection}
path_sh_local=${path_local}${job_type}${this_collection}

# Define the job that will run (load environment, save python output to log file) 
cat <<EOF > $path_sh_file.sh
#!/bin/bash
# Load the conda environment
# conda init
conda activate nnets_py38
# Run python with the specified variables 
# The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
echo "Beginning python script"
python -u $path_python ${this_collection} 2>&1 
echo 'Finished' $OAR_JOB_ID 
echo 'Finished' $OAR_JOB_ID 1>&2
EOF

# Make the job file executable
chmod +x $path_sh_file.sh

# And then execute it 
oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=16,walltime=2:00:00 -n $path_jobname --project mais 
