@echo off
pushd %~dp0

.\python-3.12.8-embed-amd64\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128