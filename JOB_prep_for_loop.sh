#!/bin/bash

#simulation='OPM026'
#for i in {2024..2068}
#for i in {2050..2068}
#simulation='OPM0263'
#for i in {1980..1981}
#for i in {1979..2018}
#for i in {1981..2018}
simulation='OPM031'
#for i in {1999..2000}
for i in {2028..2029}

do 
    year=${i}
    # Where to find the python script to run the job on 
    #path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/NN_prep_input_OPM0263.py
    #path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/NN_prep_input_OPM0263.py
    path_python=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/PREP_inputOPM.py
    # Where to save the job output
    path_jobid=/bettik/ockendeh/SCRIPTS/simpleNN_basal_melt/AIAI_scripts_notebooks/JOB_files/
    path_local=JOB_files/
    path_jobname=$path_${simulation}_${year}
    echo "Running these variables: " $path_jobname
    echo "path" $path
    path_sh_file=$path_jobid${simulation}_${year}
    path_sh_local=$path_local${simulation}_${year}
    echo "path_sh_file" $path_sh_file
    # Define the job that will run (load environment, save python output to log file) 
    cat <<EOF > $path_sh_file.sh
    #!/bin/bash
    # Load the conda environment
    # conda init
    conda activate nnets_py38
    # Run python with the specified variables 
    # The 2>&1 means that errors in the python file will appear in the stdout file not the stderr file (I think) 
    python -u $path_python ${year} 2>&1 
    echo 'Finished' $OAR_JOB_ID 
    echo 'Finished' $OAR_JOB_ID 1>&2
EOF
    
    # Make the job file executable
    chmod +x $path_sh_file.sh
    
    # And then execute it 
    oarsub -S ./$path_sh_local.sh --stdout $path_jobid/$path_jobname.o --stderr $path_jobid/$path_jobname.e -l nodes=1/core=4,walltime=2:00:00 -n $path_jobname --project mais 
    
    # And then remove the sh file which runs the code because they clutter up the folder and are all just repeats 
    #rm $path_jobname.sh
done
