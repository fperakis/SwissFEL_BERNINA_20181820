#!/bin/bash
RUNDIR=/sf/bernina/data/p17743/res/work/jobs/
LOGDIR=/sf/bernina/data/p17743/res/work/logs/
DATADIR=/sf/bernina/data/p17743/res/scan_info/

# check status of your submitted jobs:
#$ squeue -u <user_name>
# cancel submitted job:
#$ scancel <job_id>
# view information about Slurm nodes:
#$ sinfo
# request a job allocation
#$ salloc -p day
# allocate compute nodes and run a script on allocated nodes:
#$ srun <script>
# submit batch script:
#$ sbatch <script>

if [ $# -eq 0 ];
then
    echo "USAGE: ./submit_jobs.sh <run1> [<shots1> <threshold1, <run2> <shots2> <threshold2>, ...]"
else
    for (( i=1; i<=$#/3; i++ )); do
	(( j=3*i-2 ))
	eval RUN=\${$j}
	(( j=3*i-1 ))
	eval SHOTS=\${$j}
	(( j=3*i ))
	eval THRESHOLD=\${$j}
	
	if test ! -e $DATADIR/run$RUN.json; then
	    echo "ABORT! json file does not exist: "$DATADIR"run"$RUN".json"
	    echo "AVAILABLE FILES"
	    ls $DATADIR
	    exit 1
	fi

	JOB='run'$RUN'.sh'
	CURRDIR=`pwd`
	
	if test ! -e $CURRDIR/process.py; then
	    echo "ABORT! processing script does not exist in current directory: "$CURRDIR/"process.py"
	    exit 1
	fi
	
	echo "Submitting "$JOB"..."

	echo '#!/bin/bash' > $JOB
	echo '' >> $JOB
	echo '#SBATCH -o '$LOGDIR'run'$RUN'.out' >> $JOB
	echo '#SBATCH -e '$LOGDIR'run'$RUN'.err' >> $JOB
	echo '#SBATCH --partition=day' >> $JOB
	echo '' >> $JOB
	echo 'HOST=`hostname`' >> $JOB
	echo 'echo "Node: $HOST"' >> $JOB
	echo '' >> $JOB
	echo './process.py -r '$RUN' -s '$SHOTS' -t '$THRESHOLD >> $JOB
	
	sbatch $JOB
	
	mv $JOB $RUNDIR/$JOB
    done
    if [ $# -lt 3 ];
    then
	eval RUN=\${1}
	eval SHOTS=\${2}
	
	if test ! -e $DATADIR/run$RUN.json; then
	    echo "ABORT! json file does not exist: "$DATADIR"run"$RUN".json"
	    echo "AVAILABLE FILES"
	    ls $DATADIR
	    exit 1
	fi

	JOB='run'$RUN'.sh'
	CURRDIR=`pwd`
	
	if test ! -e $CURRDIR/process.py; then
	    echo "ABORT! processing script does not exist in current directory: "$CURRDIR/"process.py"
	    exit 1
	fi
	
	echo "Submitting "$JOB"..."

	echo '#!/bin/bash' > $JOB
	echo '' >> $JOB
	echo '#SBATCH -o '$LOGDIR'run'$RUN'.out' >> $JOB
	echo '#SBATCH -e '$LOGDIR'run'$RUN'.err' >> $JOB
	echo '#SBATCH --partition=day' >> $JOB
	echo '' >> $JOB
	echo 'HOST=`hostname`' >> $JOB
	echo 'echo "Node: $HOST"' >> $JOB
	echo '' >> $JOB
	if [ -z $SHOTS ]; then
	    echo './process.py -r '$RUN >> $JOB
	else
	    echo './process.py -r '$RUN' -s '$SHOTS >> $JOB
	fi

	sbatch $JOB
	
	mv $JOB $RUNDIR/$JOB
    fi
fi
