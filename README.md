# PyDRK

### Usage
#### for single files
> python3 PyDRK.py -f path/to/xd01.fco
#### or to plot multiple fco in one plot
> python3 PyDRK.py -f "path/to/xd01.fco path/to/xd02.fco path/to/xd03.fco"

### Output
The program writes a multipage pdf file in the folder _path/to/_ with the name _xd01.pdf_ or _xd01_xd02_xd03.pdf_.

### Optional arguments
 - -r [float] to adjust the DRK plot steps [default: 0.1]
 - -o [flag] to open the pdf on exit
 - -l [flag] to additionally plot the LnK+1
