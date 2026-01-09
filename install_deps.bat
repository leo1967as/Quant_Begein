@echo off
echo Installing dependencies using uv...
uv venv
call .venv\Scripts\activate
uv pip install polars pyarrow
echo Done.
pause
