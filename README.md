# Job Market Analysis and Recommendation System

Project on Upwork job postings (13 Feb - 24 Mar 2024, about 244,000 jobs).

## Folder structure
```
job_market_project/
  notebook/Job_Market_Analysis.ipynb   <- all 8 tasks (analysis, graphs, insights)
  notebook/hourly_rate_map.html        <- interactive map of hourly rates (Task 4)
  notebook/Dockerfile                  <- Jupyter service
  app/                                 <- Flask app (Task 5 engine + Task 6 dashboard)
    app.py, jobdata.py, templates/, requirements.txt, Dockerfile
  nginx/                               <- web server in front of Flask (Dockerfile + config)
  data/upwork_job.csv                  <- dataset
  docker-compose.yml                   <- runs all services together
  API_DOCS.md                          <- API documentation
```

## 1. Run the notebook
Needs Python 3.10+ with: `pandas numpy scipy scikit-learn matplotlib seaborn plotly jupyterlab`
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly jupyterlab
cd notebook
jupyter lab
```
Open `Job_Market_Analysis.ipynb` and run all cells. The notebook finds the CSV in `../data/`.
It takes a few minutes (the keyword and TF-IDF steps are the slowest).

## 2. Run the Flask app without Docker
```bash
cd app
pip install -r requirements.txt
set DATA_PATH=../data/upwork_job.csv        # Windows (use "export DATA_PATH=..." on Mac/Linux)
python app.py
```
Open http://localhost:5000 (recommendation page) and http://localhost:5000/dashboard.
Start-up takes about 20 seconds because the data is loaded and the model is built.

## 3. Run everything with Docker
Docker and Docker Compose are required.

**Build and run**
```bash
docker compose build
docker compose up -d
```

| Service | URL | What it is |
|---|---|---|
| web (nginx) | http://localhost | main entry, sends requests to the app |
| app (Flask + gunicorn) | internal, port 5000 | recommendation API + dashboard |
| notebook (Jupyter) | http://localhost:8888 (token: `jobmarket`) | run the analysis notebook |

Check the logs / stop:
```bash
docker compose logs -f app
docker compose down
```

**Monthly update of the dashboard**
1. Replace / append the new data in `data/upwork_job.csv`.
2. Reload the app data without a restart: `curl -X POST http://localhost/api/refresh`

**Deploy on a server**
1. Install Docker on the server and copy this folder to it.
2. Run `docker compose up -d --build`.
3. Open port 80 in the firewall. For HTTPS put the server behind a domain with a reverse proxy / certificate (for example Let's Encrypt).
4. For safety do not expose port 8888 on a public server (remove the `notebook` service from `docker-compose.yml` or protect it).

## API
See `API_DOCS.md`.
