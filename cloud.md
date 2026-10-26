# Cloud Deployment & AWS S3 / Floci Architecture Guide

This comprehensive guide details the **cloud architecture, workflow, and deployment steps** for the **Chest X-Ray Disease Detection System** (FastAPI backend + PyTorch deep learning models + React frontend + AWS S3 cloud storage).

1. [Architecture & Workflow: React -> AWS S3 -> PyTorch -> Frontend](#1-architecture--end-to-end-workflow)
2. [Local AWS S3 Testing with Floci (Zero-Cost Emulator)](#2-local-aws-testing-with-floci)
3. [AWS S3 Bucket & IAM Configuration in the Cloud](#3-aws-s3-bucket--iam-setup)
4. [AWS EC2 Deployment Guide (Free Tier)](#4-aws-ec2-deployment-12-months-free)
5. [Docker Compose Full-Stack Deployment (With Floci)](#5-docker-compose-full-stack-deployment)
6. [Alternative Cloud Options (GCP / Zoho Catalyst / Hugging Face)](#6-alternative-cloud-platforms)

---

## 1. Architecture & End-to-End Workflow

```
┌─────────────────┐       1. Upload X-Ray Image       ┌────────────────────────┐
│  React Frontend │ ────────────────────────────────> │     FastAPI Server     │
│  (Port 3000/80) │ <──────────────────────────────── │      (Port 8000)       │
└─────────────────┘       4. Results & Heatmaps       └───────────┬────────────┘
                                                                  │
                                      ┌───────────────────────────┴───────────────────────────┐
                                      ▼                                                       ▼
                       2. Archive Original Image                               3. Deep Learning Inference
                       ┌───────────────────────┐                               ┌───────────────────────┐
                       │     AWS S3 Bucket     │                               │ 4 PyTorch DL Models:  │
                       │          OR           │                               │ • DenseNet-121 Core   │
                       │ Floci Emulator (:4566)│                               │ • COVID-19 Specialist │
                       └───────────────────────┘                               │ • Pneumonia Specialist│
                                                                               │ • TB Specialist       │
                                                                               │ + Grad-CAM Heatmaps   │
                                                                               └───────────────────────┘
```

### Detailed Workflow Stages:

1. **Upload & Pre-flight Validation (Frontend)**:
   - The user selects or drags a chest X-ray image (PNG/JPG/DICOM).
   - The React frontend validates file format and displays an instant local preview.
2. **S3 Object Archiving (Storage)**:
   - FastAPI receives the multipart image payload.
   - Using the AWS Boto3 SDK, the image is archived into an **AWS S3 bucket** (e.g. `chest-xray-images/xrays/YYYY/MM/timestamp_uuid_filename.png`) with clean metadata.
   - When running locally, Boto3 connects seamlessly to **Floci** (`http://localhost:4566`), creating the bucket and object with zero cloud cost.
   - In production, Boto3 writes directly to the real AWS S3 Cloud bucket.
   - An authenticated/presigned URL is generated for clinical verification.
3. **Multi-Model Inference & Explainability (Backend)**:
   - The image is passed through 4 deep learning models:
     - **Main Classifier**: `torchxrayvision` DenseNet-121 (4-class: COVID-19, Normal, Pneumonia, Tuberculosis).
     - **3 Disease Specialists**: ResNet-18 binary classifiers.
   - **Grad-CAM Attention Heatmap**: Computes gradient activations across the final convolutional layers to highlight anomalous pulmonary regions.
4. **Displaying Results (Frontend)**:
   - The frontend receives the consensus diagnosis, class probabilities, confidence score, recommendation, Grad-CAM heatmap, and the **S3 Storage Confirmation card** (Bucket name, S3 Key, Region, and direct S3 link).

---

## 2. Local AWS Testing with Floci

[Floci](https://github.com/floci-io/floci) is a free, open-source, ultra-fast local AWS emulator built with Quarkus and GraalVM (~24ms startup time, ~13MB memory footprint). It provides full AWS-compatible APIs (including S3) on port `4566` without requiring AWS credentials or credit cards.

### Starting Floci Locally:

**Using Docker:**
```bash
docker run -d -p 4566:4566 --name floci floci/floci:latest
```

**Using the Floci CLI:**
```bash
# Install floci-cli
curl -fsSL https://floci.io/install.sh | sh

# Start emulator
floci start
eval $(floci env)
```

**Environment Variables for Floci:**
```ini
FLOCI_ENDPOINT_URL=http://localhost:4566
AWS_S3_BUCKET_NAME=chest-xray-images
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
ENABLE_S3_STORAGE=true
```

---

## 3. AWS S3 Bucket & IAM Setup

When deploying to real AWS Cloud:

### 3.1 Create the S3 Bucket
```bash
aws s3 mb s3://chest-xray-images --region us-east-1
```

### 3.2 S3 Bucket CORS Configuration
Under **S3** > **chest-xray-images** > **Permissions** > **Cross-origin resource sharing (CORS)**:
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST", "HEAD"],
    "AllowedOrigins": ["*"],
    "ExposeHeaders": ["ETag"]
  }
]
```

### 3.3 IAM Policy for EC2 / App
Attach an IAM policy granting S3 permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:HeadBucket"
      ],
      "Resource": [
        "arn:aws:s3:::chest-xray-images",
        "arn:aws:s3:::chest-xray-images/*"
      ]
    }
  ]
}
```

---

## 4. AWS EC2 Deployment (12 Months Free)


AWS offers 750 hours/month of a `t2.micro` (or `t3.micro` in eligible regions) instance for 12 months under the Free Tier.

### Step 1.1: Launch an EC2 Instance
1. Sign in to the [AWS Management Console](https://aws.amazon.com/console/).
2. Navigate to **EC2** > **Instances** > **Launch Instances**.
3. Configure the instance:
   - **Name**: `chest-xray-app`
   - **OS Image (AMI)**: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type** (Free tier eligible).
   - **Instance Type**: `t2.micro` (1 vCPU, 1 GiB Memory).
   - **Key Pair**: Click *Create new key pair*, name it (e.g., `chest-xray-key`), and download the `.pem` file.
   - **Network Settings**:
     - Check **Allow SSH traffic from Anywhere** (or My IP).
     - Check **Allow HTTP traffic from the internet**.
     - Check **Allow HTTPS traffic from the internet**.
   - **Storage**: Change volume size from 8 GiB to **20 GiB or 30 GiB** (AWS Free Tier includes up to 30 GiB of gp3/gp2 storage).
4. Click **Launch Instance**.

### Step 1.2: Connect via SSH
Open PowerShell or your terminal where the `.pem` file is saved:

```bash
# On Linux / macOS / WSL:
chmod 400 chest-xray-key.pem
ssh -i "chest-xray-key.pem" ubuntu@<YOUR-EC2-PUBLIC-IP>

# On Windows PowerShell:
ssh -i .\chest-xray-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

### Step 1.3: Enable Swap Memory (CRITICAL for 1 GB RAM)
Run these commands on the server to add 3 GB of virtual swap memory so PyTorch does not run out of memory:

```bash
sudo fallocate -l 3G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

### Step 1.4: Install System Dependencies, Python, and Node.js
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git nginx libgl1 libglib2.0-0 curl

# Install Node.js 20 LTS
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

### Step 1.5: Clone Project and Install Python Requirements
```bash
git clone <YOUR-GITHUB-REPO-URL> ~/respiratory-x-ray
cd ~/respiratory-x-ray

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

*(Ensure the `.pth` model files are inside `~/respiratory-x-ray/backend/models/` and root via Git LFS or `scp`).*

### Step 1.6: Build the Frontend
```bash
cd ~/respiratory-x-ray/frontend
npm install
npm run build
```
*(The production build will be generated in `frontend/dist`).*

### Step 1.7: Configure Systemd Service for Backend
Create a service to keep FastAPI running automatically in the background:

```bash
sudo nano /etc/systemd/system/chest-xray-backend.service
```

Paste the following:

```ini
[Unit]
Description=Chest X-Ray FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/respiratory-x-ray
ExecStart=/home/ubuntu/respiratory-x-ray/.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always

[Install]
WantedBy=multi-user.target
```

Start and enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl start chest-xray-backend
sudo systemctl enable chest-xray-backend
sudo systemctl status chest-xray-backend
```

### Step 1.8: Configure Nginx Reverse Proxy
Route web traffic so port 80 serves the React frontend and forwards `/api/` calls to FastAPI:

```bash
sudo nano /etc/nginx/sites-available/chest-xray
```

Paste the configuration:

```nginx
server {
    listen 80;
    server_name _;

    # Frontend React Static Files
    location / {
        root /home/ubuntu/respiratory-x-ray/frontend/dist;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    # Backend API Reverse Proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 25M;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
    }

    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
    }
}
```

Enable the site and reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/chest-xray /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

**Done!** Open your browser and go to `http://<YOUR-EC2-PUBLIC-IP>` to use the app.

---

## 2. Google Cloud Platform (GCP) Deployment (Always Free)

Google Cloud offers an **e2-micro instance** with 30 GB standard persistent disk in `us-central1`, `us-west1`, or `us-east1` that is **Always Free**.

### Step 2.1: Create Compute Engine VM
1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Select or create a project.
3. Open **Compute Engine** > **VM instances** > **Create Instance**.
4. Set:
   - **Name**: `chest-xray-vm`
   - **Region**: `us-central1` (Iowa), `us-east1` (South Carolina), or `us-west1` (Oregon) *(Must pick one of these for Always Free tier)*.
   - **Series**: `E2`
   - **Machine Type**: `e2-micro` (2 vCPUs, 1 GB Memory).
   - **Boot disk**: Click *Change*, choose **Ubuntu 22.04 LTS**, Disk type: **Standard persistent disk**, Size: **30 GB** (Always free limit).
   - **Firewall**: Check both **Allow HTTP traffic** and **Allow HTTPS traffic**.
5. Click **Create**.

### Step 2.2: Connect and Setup
Click the **SSH** button next to your VM instance in the Google Cloud Console.

1. **Setup 3GB Swap** (Mandatory for PyTorch on e2-micro):
   ```bash
   sudo fallocate -l 3G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

2. **Install Packages, Clone Repo & Run**:
   Follow the exact same steps as **AWS Steps 1.4 to 1.8** above.
3. Access your app at `http://<YOUR-GCP-EXTERNAL-IP>`.

---

## 3. Zoho Catalyst Deployment (Free Tier)

**Zoho Catalyst** is Zoho's developer cloud platform. It offers a generous **Free Tier (Spark plan)** that includes:
- **AppSail** (Custom containers / web servers for Python/FastAPI)
- **Web Client Hosting** (Static hosting for React)
- Free compute credits monthly

### Step 3.1: Install Catalyst CLI
Install the official Zoho Catalyst CLI on your local computer:

```bash
npm install -g zcatalyst-cli
```

### Step 3.2: Log in to Zoho Catalyst
```bash
catalyst login
```
*(A browser window opens to log in with your Zoho account).*

### Step 3.3: Initialize Project
Inside your project root:

```bash
cd "c:\Users\Nagineni Dhanush\Downloads\respiratory-x-ray-main"
catalyst init
```
1. Select your Catalyst Project.
2. Select **AppSail** (for the FastAPI backend) and **Client** (for the React frontend).

### Step 3.4: Configure AppSail for the Backend
In the created `app-config.json` inside your AppSail directory:

```json
{
  "command": "uvicorn backend.main:app --host 0.0.0.0 --port 8080",
  "build_path": "",
  "stack": "python3.10",
  "memory": 1024
}
```

### Step 3.5: Build and Deploy
1. Build the React frontend:
   ```bash
   cd frontend
   npm run build
   ```
2. Point your client directory in `catalyst.json` to `frontend/dist`.
3. Deploy to Zoho Catalyst Cloud:
   ```bash
   catalyst deploy
   ```
Zoho will provide you with a live `*.catalystserverless.com` URL.

---

## 4. Recommended Alternative: Hugging Face + Vercel (Zero Cost)

Because PyTorch inference on deep learning models is CPU/RAM intensive, the easiest and most generous **100% free** setup is:

| Component | Host | Free Specs | Cost |
| :--- | :--- | :--- | :--- |
| **Backend (FastAPI + PyTorch)** | **Hugging Face Spaces** | **16 GB RAM**, 2 vCPU, Git LFS model hosting | **$0 / month** |
| **Frontend (React + Vite)** | **Vercel** or **Netlify** | Fast global edge CDN, Automatic SSL | **$0 / month** |

### Step 4.1: Deploy Backend to Hugging Face Spaces (16 GB Free RAM)
1. Sign up at [huggingface.co](https://huggingface.co).
2. Click **New Space** > Choose **Docker** (Blank) > Hardware: **Free (16 GB RAM)**.
3. In your space, create a `Dockerfile`:
   ```dockerfile
   FROM python:3.10-slim

   WORKDIR /app

   RUN apt-get update && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
       curl \
       && rm -rf /var/lib/apt/lists/*

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY backend/ ./backend/
   COPY *.pth ./

   EXPOSE 7860

   CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
4. Push your code and `.pth` model weights.
5. Your public backend endpoint:
   `https://<your-username>-chest-xray-api.hf.space`

### Step 4.2: Deploy Frontend to Vercel
1. Go to [vercel.com](https://vercel.com) and click **Add New** > **Project**.
2. Select your repository.
3. Set **Root Directory** to `frontend`.
4. Add Environment Variable:
   - Name: `VITE_API_BASE_URL`
   - Value: `https://<your-username>-chest-xray-api.hf.space`
5. Click **Deploy**. Done!

---

## 5. Handling Model Weights (`.pth` Files)

Standard GitHub repositories limit individual file uploads to **100 MB**. If your model weights exceed this (e.g. `best_model_balanced.pth` ~94 MB, DenseNet ~28 MB, ResNets ~45 MB each):

1. **Use Git LFS (Large File Storage)**:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git commit -m "Track model checkpoints with Git LFS"
   ```

2. **Direct Server Transfer (`scp`)**:
   Instead of uploading models to GitHub, you can securely copy model weights directly from your local computer to your cloud VM:
   ```bash
   scp -i chest-xray-key.pem *.pth ubuntu@<YOUR-EC2-PUBLIC-IP>:~/respiratory-x-ray/
   ```

---

## Summary Checklist

- [x] Enable 3 GB Swap space on AWS `t2.micro` or GCP `e2-micro` instances.
- [x] Build frontend production bundle (`npm run build`).
- [x] Use Nginx to serve frontend static files and proxy `/api/` calls to FastAPI on port 8000.
- [x] Configure systemd to keep backend running 24/7.
- [x] (Optional) Secure with free HTTPS using Let's Encrypt: `sudo certbot --nginx`.
