FROM python:3.10.18-slim

RUN pip install uv

WORKDIR /app

COPY . .

# we use custom requirements.txt instead of pyproject.toml
# because it was easier to setup and sync dependencies so far
RUN uv pip install --system -r requirements.txt

EXPOSE 8888

CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]