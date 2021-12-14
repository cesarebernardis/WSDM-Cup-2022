import numpy, os, glob

from setuptools import Command, Extension, setup

from Cython.Build import cythonize
from Cython.Compiler.Options import _directive_defaults

_directive_defaults['linetrace'] = True
_directive_defaults['binding'] = True
compile_args = ['-O3', '-ffast-math', '-fopenmp', '-Wmaybe-uninitialized', '-Wsign-compare']
link_args = ['-fopenmp']

# Look for cython files to compile
cython_files = []
for path in glob.glob("RecSysFramework/**/*.pyx", recursive=True):
    cython_files.append((path.replace(".pyx", "").replace(os.sep, "."), path))


modules = [Extension(file[0], [file[1]], language='c++', extra_compile_args=compile_args, extra_link_args=link_args)
           for file in cython_files]
setup(
    name='RecSysFramework',
    version="1.0.0",
    description='Recommender System Framework',
    url='https://github.com/MaurizioFD/RecSysFramework.git',
    author='Cesare Bernardis, Maurizio Ferrari Dacrema',
    author_email='cesare.bernardis@polimi.it, maurizio.ferrari@polimi.it',
    install_requires=['numpy>=0.14.0',
                      'pandas>=0.22.0',
                      'scipy>=0.16',
                      'scikit-learn>=0.19.1',
                      'matplotlib>=3.0',
                      'Cython>=0.27',
                      'nltk>=3.2.5',
                      'tqdm>=4.59',
                      'similaripy>=0.0.11'],
    packages=['RecSysFramework',
              'RecSysFramework.Evaluation',
              'RecSysFramework.DataManager',
              'RecSysFramework.DataManager.Reader',
              'RecSysFramework.DataManager.DatasetPostprocessing',
              'RecSysFramework.DataManager.Splitter',
              'RecSysFramework.Recommender',
              'RecSysFramework.Recommender.GraphBased',
              'RecSysFramework.Recommender.KNN',
              'RecSysFramework.Recommender.MatrixFactorization',
              'RecSysFramework.Recommender.SLIM',
              'RecSysFramework.Recommender.SLIM.BPR',
              'RecSysFramework.Recommender.SLIM.ElasticNet',
              'RecSysFramework.ParameterTuning',
              'RecSysFramework.Utils',
              'RecSysFramework.Utils.Similarity',
              ],
    setup_requires=["Cython >= 0.27"],
    ext_modules=cythonize(modules),
    include_dirs=[numpy.get_include()]
)
