#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, re, argparse, subprocess
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Qt5Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def init_argparser():
    parser = argparse.ArgumentParser(description='Find local maxima on frames, filter peaks and fit the resulting reflection profiles assuming a Gaussian model to estimate the peak flux.')
    parser.add_argument('-f', required=True,  default='',  metavar='path',  type=str,   dest='_FILE', help='.fco file')
    parser.add_argument('-r', required=False, default=0.1, metavar='float', type=float, dest='_STEP', help='DRK plot steps')
    parser.add_argument('-o', required=False, action='store_true',  dest='_OPEN', help='Open pdf on exit')
    parser.add_argument('-l', required=False, action='store_true',  dest='_PLNK', help='Plot LnK+1')
    # print help if script is run without arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        raise SystemExit
    return parser.parse_args()

def read_fco(fname):
    data = np.genfromtxt(fname, skip_header=26, usecols=(0,1,2,3,4,5,6,7))
    used = data[data[::,7] == 0]
    return used
    
def main():
    mpl.rcParams['figure.figsize'] = [12.60, 7.68]
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['figure.titlesize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['savefig.directory'] = os.getcwd()
    
    _ARGS = init_argparser()
    flist = _ARGS._FILE.split()
    pdf_path = os.path.dirname(os.path.abspath(flist[0]))
    pdf_name = '_'.join(np.asarray(list(map(os.path.splitext, flist)))[:,0])
    pdf_file = os.path.join(pdf_path, pdf_name + '.pdf')
    
    dlist = []
    colors = {0:'#003d73', 1:'#37a0cb', 2:'#00aba4',
              3:'#ee7f00', 4:'#e2007a', 5:'#8bad3f',
              6:'#fabb00', 7:'#655a9f', 8:'#e2001a'}
    
    markers = {0:'o', 1:'^', 2:'s', 3:'D', 4:'v'}
    
    if len(flist) > len(colors) * len(markers):
        print('Too many files!')
        raise SystemExit
    
    for f in flist:
        data = read_fco(f)
        Ic = data[:,3]
        Io = data[:,4]
        Is = data[:,5]
        stl = data[:,6]
        dlist.append((os.path.basename(f), Ic, Io, Is, stl))
    
    with PdfPages(pdf_file) as pdf:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10, wspace=0.1, hspace=0.1)
    
        # NORMPROB
        for idx, vals in enumerate(dlist):
            Name, Ic, Io, Is, stl = vals
            x = (Ic-Io)/Is
            osm = stats.probplot(x)[0]
            ax.plot(osm[0], osm[1], marker=markers[idx//len(colors)], ls='',
                    color=colors[idx-idx//len(colors)*len(colors)],
                    alpha=0.75, ms=5, markevery=0.01,
                    zorder=3, label=Name)
        ax.plot([xdiag for xdiag in range(-20, 20)],
                [ydiag for ydiag in range(-20, 20)],
                '-', color='#878787', lw=2, alpha=1.0, zorder=1)
        ax.axhline(color='k', lw=1, alpha=0.5, zorder=1)
        ax.axvline(color='k', lw=1, alpha=0.5, zorder=1)
        ax.set_ylim(-4, 4)
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r'$\mathit{Expected\ DR}$')
        ax.set_ylabel(r'$\mathit{Experimental\ DR}$')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        pdf.savefig()
        ax.clear()
        
        # DRK
        max_x = []
        for idx, vals in enumerate(dlist):
            Name, Ic, Io, Is, stl = vals
            x = []
            y = []
            inc = _ARGS._STEP
            m = np.max(stl)
            max_x.append(m)
            for i in np.arange(0.0, m, inc):
                cond = (stl <= i) & (stl > i-inc)
                if np.sum(Ic[cond]) > 0:
                    y.append(np.sum(Io[cond])/np.sum(Ic[cond]))
                    x.append(i)
            ax.plot(x, y, marker=markers[idx//len(colors)], ls='',
                    color=colors[idx-idx//len(colors)*len(colors)],
                    ms=5, alpha=1.0, zorder=2, label=Name)
        ax.set_xlim([-0.05, max(max_x)+0.05])
        ax.set_ylim([0.8,1.2])
        ax.axhline(0.95, 0.0, color='r', lw=1, alpha=0.5, zorder=1)
        ax.axhline(1.05, 0.0, color='r', lw=1, alpha=0.5, zorder=1)
        ax.axhline(1.00, 0.0, color='#878787', lw=2, alpha=1.0, zorder=1)
        ax.set_xlabel(r'$\mathit{sin(\theta)\ /\ \lambda}$')
        ax.set_ylabel(r'$\mathit{\sum F(obs)^2\ /\ \sum F(calc)^2}$')
        ax.legend()
        ax.grid()
        ax.set_axisbelow(True)
        pdf.savefig()
        ax.clear()
        
        # LnK+1
        if _ARGS._PLNK:
            max_x = []
            for idx, vals in enumerate(dlist):
                Name, Ic, Io, Is, stl = vals
                y = np.log(Io/Ic) +1
                x = stl
                max_x.append(np.max(stl))
                ax.plot(x, y, marker=markers[idx//len(colors)], ls='',
                        color=colors[idx-idx//len(colors)*len(colors)],
                        ms=5, alpha=0.75, zorder=1, label=Name)
            ax.axhline(1.0, 0.0, color='#878787', lw=1, alpha=1.0, zorder=2)
            ax.set_xlim([0.0, max(max_x)+0.05])
            ax.set_xlabel(r'$\mathit{sin(\theta)\ /\ \lambda}$')
            ax.set_ylabel(r'$\mathit{ln[F(obs)^2\ /\ F(calc)^2]+1}$')
            ax.legend()
            ax.grid()
            ax.set_axisbelow(True)
            pdf.savefig()
            ax.clear()
    
    if _ARGS._OPEN:
        subprocess.Popen([pdf_file],shell=True)
    
if __name__ == '__main__':
    main()
