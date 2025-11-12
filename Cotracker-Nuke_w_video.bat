REM Fix OpenBLAS warning for high-core count systems (32 cores, 64 threads)
rem  set OPENBLAS_NUM_THREADS=32
rem  set MKL_NUM_THREADS=32
rem  set NUMEXPR_NUM_THREADS=32


.\python-3.12.8-embed-amd64\python.exe cotracker_nuke_app.py --video ".\assets\Assets_250929-165402_Cat_00001.mp4"