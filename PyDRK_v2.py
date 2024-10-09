#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, argparse, subprocess
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

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

def init_argparser():
    parser = argparse.ArgumentParser(description='Find local maxima on frames, filter peaks and fit the resulting reflection profiles assuming a Gaussian model to estimate the peak flux.')
    parser.add_argument('-f', required=True,  default='',  metavar='path',  type=str,   dest='_FILE', help='.fco file')
    parser.add_argument('-r', required=False, default=0.1, metavar='float', type=float, dest='_STEP', help='DRK plot steps')
    parser.add_argument('-x', required=False, action='store_true',  dest='_OPEN', help='Open pdf on exit')
    parser.add_argument('-n', required=False, action='store_true',  dest='_PNOR', help='Plot normal propability')
    #parser.add_argument('-d', required=False, action='store_true',  dest='_PDRK', help='Plot DRK')
    parser.add_argument('-l', required=False, action='store_true',  dest='_PLNK', help='Plot LnK+1')
    parser.add_argument('-o', required=False, default=0.05, metavar='float', type=float, dest='_OSET', help='DRK plot offset')
    # print help if script is run without arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        raise SystemExit
    return parser.parse_args()

def read_fco(fname):
    #  Ic = data[:,3]
    #  Io = data[:,4]
    #  Is = data[:,5]
    #  stl = data[:,6]
    #
    _, ext = os.path.splitext(fname)
    if ext == '.fcf':
        with open(fname) as of:
            lines = of.readlines()
        ucp = list()
        hdr_len = 0
        code = 0
        ucp = []
        for line in lines:
            hdr_len += 1
            if '_shelx_refln_list_code' in line:
                code = int(line.split()[1])
            elif '_cell_' in line:
                ucp.append(line.split()[1])
            elif '_refln_observed_status' in line:
                ucp = list(map(float, ucp))
                if code > 0 and len(ucp) == 6:
                    break
                else:
                    print('Error reading fcf!')
                raise SystemExit
        data = np.genfromtxt(fname, skip_header=hdr_len, usecols=(0,1,2,3,4,5))
        if code == 6:
            # _refln_F_squared_meas
            # _refln_F_squared_sigma
            # _refln_F_calc
            data[:,5] = data[:,5]**2
            data.swapaxes(5, 3)
            data.swapaxes(5, 4)
        if code == 4:
            # _refln_F_squared_calc
            # _refln_F_squared_meas
            # _refln_F_squared_sigma
            pass
        stl = calc_stl(ucp, data[:,:3]).reshape(-1,1)
        used = np.hstack([data, stl])
    elif ext == '.fco':
        data = np.genfromtxt(fname, skip_header=26, usecols=(0,1,2,3,4,5,6,7))
        used = data[data[::,7] == 0]
    return used

def calc_stl(cell, hkl):
    """
    Calculate resolution in d-spacing (dsp) and sin(theta)/lambda (stl)
    :param ucp: Unit cell parameters, list, [a, b, c, alpha, beta, gamma]
    :param hkl: 2d array with [h,k,l]
    :return: array of [stl]
    """
    def cart_from_cell(cell):
        """
        Calculate a,b,c vectors in cartesian system from lattice constants.
        :param cell: a,b,c,alpha,beta,gamma lattice constants.
        :return: a, b, c vector
        """
        if cell.shape != (6,):
            raise ValueError('Lattice constants must be 1d array with 6 elements')
        a, b, c = cell[:3] * 1E-10
        alpha, beta, gamma = np.radians(cell[3:])
        av = np.array([a, 0, 0], dtype=float)
        bv = np.array([b * np.cos(gamma), b * np.sin(gamma), 0], dtype=float)
        # calculate vector c
        x = np.cos(beta)
        y = (np.cos(alpha) - x * np.cos(gamma)) / np.sin(gamma)
        z = np.sqrt(1. - x**2. - y**2.)
        cv = np.array([x, y, z], dtype=float)
        cv /= np.linalg.norm(cv)
        cv *= c
        return av, bv, cv
    
    def matrix_from_cell(cell):
        """
        Calculate transform matrix from lattice constants.
        :param cell: a,b,c,alpha,beta,gamma lattice constants in
                                angstrom and degree.
        :param lattice_type: lattice type: P, A, B, C, H
        :return: transform matrix A = [a*, b*, c*]
        """
        cell = np.array(cell)
        av, bv, cv = cart_from_cell(cell)
        a_star = (np.cross(bv, cv)) / (np.cross(bv, cv).dot(av))
        b_star = (np.cross(cv, av)) / (np.cross(cv, av).dot(bv))
        c_star = (np.cross(av, bv)) / (np.cross(av, bv).dot(cv))
        A = np.zeros((3, 3), dtype='float')  # transform matrix
        A[:, 0] = a_star
        A[:, 1] = b_star
        A[:, 2] = c_star
        return np.round(A,6)

    A = matrix_from_cell(cell)
    # calculate the d-spacing
    # go from meters to Angstrom
    dsp = 1/np.linalg.norm(A.dot(hkl.T).T, axis=1) * 1E10
    stl = 1/(2*dsp)
    return stl

def main():
    _ARGS = init_argparser()
    _ARGS._PDRK = True
    flist = _ARGS._FILE.split()
    pdf_path = os.path.dirname(os.path.abspath(flist[0]))
    #pdf_name = '_'.join(np.asarray(list(map(os.path.splitext, flist)))[:,0])
    #pdf_file = os.path.join(pdf_path, pdf_name + '.pdf')
    pdf_file = os.path.join(pdf_path, 'pyDRK.pdf')
    
    dlist = []
    colors = {0:'#003d73', 3:'#37a0cb', 2:'#00aba4',
              4:'#ee7f00', 1:'#e2007a', 5:'#8bad3f',
              8:'#fabb00', 6:'#655a9f', 7:'#e2001a'}
    
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
        if _ARGS._PNOR:
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
        if _ARGS._PDRK:
            max_x = []
            for idx, vals in enumerate(dlist):
                Name, Ic, Io, Is, stl = vals
                x = []
                y = []
                inc = _ARGS._STEP
                offset = inc * _ARGS._OSET
                m = np.max(stl)
                max_x.append(m)
                for i in np.arange(0.0, m, inc):
                    cond = (stl <= i) & (stl > i-inc)
                    if np.sum(Ic[cond]) > 0:
                        y.append(np.sum(Io[cond])/np.sum(Ic[cond]))
                        x.append(i+idx*offset)
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
