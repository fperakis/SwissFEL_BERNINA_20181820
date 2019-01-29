# SwissFEL_BERNINA_20181820

Data analysis repository for experiments at SwissFEL between Jan 27- Feb 4, 2019 (proposal number: 20181820)

Inlcudes code to:
- read data from .json files
- do angular integration
- calculate pump-probe signal

-----------------------------
Instalation on an SwissFEL machine
1) ssh to a data analysis node by:
```bash
$ salloc -p day   # allocate nodes
$ squeue          # to view which node is allocated
$ ssh -X ra-c-XXX # where XXX is the node number (for ex. 006)
```

2) activate bernina conda environment
```bash
$ source /sf/bernina/bin/anaconda_env
```

note that your ~/.bashrc should contain:

```bash
# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

#Setup the environment for the PSI pmodules
#idocumentation : https://amas.psi.ch/Pmodules
if [ -f /etc/scripts/pmodules.sh ]; then
        . /etc/scripts/pmodules.sh
fi
```
 

3) download the repo:

```bash
$ git clone https://github.com/fperakis/SwissFEL_BERNINA_20181820.git
```
