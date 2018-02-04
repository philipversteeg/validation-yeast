""" Package settings file"""
import os

## Settings 
_name = 'Prediction'                                            # printable name
verbose = 1                                                     # default global level of print output
server = os.sys.platform != 'darwin'                            # true if on linux or windows
save_task_threshold= 10.                                        # for each prediction task, a

## Package folders
_folder_package = os.path.abspath(os.path.dirname(__file__))    # package folder
_folder_prediction = _folder_package + '/prediction'
_folder_extern_R = _folder_prediction + '/extern/R'             # external R folder

## Project folders 
folder = '..'                                                   # base folder for computation
folder_test = folder + '/test'                                  # containts (unit)tests and output
folder_results = folder + '/results'                             # result base folder
# create folders if not exists
if not os.path.exists(folder_results): os.makedirs(folder_results)

## Parallel
ncores_total = os.sysconf('SC_NPROCESSORS_ONLN')                # number of cores in total, default is half of system threads
ncores_method = int(ncores_total / 2)                           # number of cores that each multiprocessed method can use

## Matplotlib settings
# import matplotlib 2.1 default parameters to older version
plt_rcParams = {
    'axes.color_cycle' : ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'axes.linewidth' : 1.0,
    'figure.dpi' : 100,
    'figure.facecolor' : 'w',
    'figure.figsize' : [6.4, 4.8],
    'figure.subplot.bottom' : 0.11,
    'figure.subplot.top' : 0.9,
    'font.cursive' : ['Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT', 'Felipa', 'cursive'],
    'font.monospace' : ['DejaVu Sans Mono', 'Bitstream Vera Sans Mono', 'Computer Modern Typewriter', 'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace'],
    'font.sans-serif' : ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif'],
    'font.serif' : ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook', 'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times New Roman', 'Times', 'Palatino', 'Charter', 'serif'],
    'font.size' : 10.0,
    'grid.color' : 'b0b0b0',
    'grid.linestyle' : '-',
    'grid.linewidth' : 0.5,
    'image.interpolation' : 'nearest',
    'image.resample' : True,
    'legend.fancybox' : True,
    'legend.fontsize' : 'medium',
    'legend.loc' : 'best',
    'legend.numpoints' : 1,
    'lines.linewidth' : 1.5,
    'lines.markeredgewidth' : 1.0,
    'mathtext.bf' : 'sans:bold',
    'mathtext.it' : 'sans:italic',
    'mathtext.rm' : 'sans',
    'mathtext.sf' : 'sans',
    'pgf.preamble' : [],
    'text.hinting' : 'auto',
    'text.latex.preamble' : [],
    # 'xtick.direction' : 'out',
    'xtick.major.pad' : 3.5,
    'xtick.major.size' : 3.5,
    'xtick.major.width' : 0.8,
    'xtick.minor.pad' : 3.4,
    'xtick.minor.width' : 0.6,
    # 'ytick.direction' : 'out',
    'ytick.major.pad' : 3.5,
    'ytick.major.size' : 3.5,
    'ytick.major.width' : 0.8,
    'ytick.minor.pad' : 3.4,
    'ytick.minor.width' : 0.6
}