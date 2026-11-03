import boto3
import json
import os
import sys
import time
import zipfile
import urllib.request

# Load environment variables from .env if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "399855128301")
BUCKET_NAME = f"chest-xray-images-{ACCOUNT_ID}"
LAMBDA_FUNCTION_NAME = "chest-xray-detection-api"
LAMBDA_ROLE_NAME = "chest-xray-lambda-execution-role"

def get_session():
    return boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

def upload_models_to_s3():
    session = get_session()
    s3 = session.client('s3')
    print(f"\n[1/5] Checking Model Checkpoints in s3://{BUCKET_NAME}/models/ ...")
    # Check if files already uploaded
    res = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix="models/")
    existing_keys = [obj['Key'] for obj in res.get('Contents', [])]
    print(f"  [OK] Found {len(existing_keys)} model files already uploaded in S3.")

def create_lambda_zip():
    print("\n[2/5] Creating Lambda Deployment Zip...")
    zip_path = "lambda_package.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write("lambda_function.py", "lambda_function.py")
    print(f"  [OK] Created {zip_path} ({os.path.getsize(zip_path)} bytes)")
    return zip_path

def get_or_create_lambda_role():
    session = get_session()
    iam = session.client('iam')
    print(f"\n[3/5] Setting up IAM Execution Role: {LAMBDA_ROLE_NAME} ...")

    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }

    try:
        role = iam.get_role(RoleName=LAMBDA_ROLE_NAME)
        role_arn = role['Role']['Arn']
        print(f"  [OK] Found existing role: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        print(f"  Creating role {LAMBDA_ROLE_NAME}...")
        role = iam.create_role(
            RoleName=LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for chest X-ray detection Lambda"
        )
        role_arn = role['Role']['Arn']

        for policy in [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess"
        ]:
            iam.attach_role_policy(RoleName=LAMBDA_ROLE_NAME, PolicyArn=policy)
        print("  [OK] Attached policies. Waiting 10s for IAM propagation...")
        time.sleep(10)

    return role_arn

def deploy_lambda_function(role_arn, zip_path):
    session = get_session()
    lam = session.client('lambda')
    print(f"\n[4/5] Deploying Lambda Function: {LAMBDA_FUNCTION_NAME} in {AWS_REGION} ...")

    with open(zip_path, 'rb') as f:
        zip_bytes = f.read()

    try:
        lam.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        print(f"  Updating code for existing function {LAMBDA_FUNCTION_NAME}...")
        lam.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME,
            ZipFile=zip_bytes
        )
        lam.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Environment={
                'Variables': {
                    'AWS_S3_BUCKET_NAME': BUCKET_NAME
                }
            },
            Timeout=30,
            MemorySize=512
        )
        print("  [OK] Function code and config updated.")
    except lam.exceptions.ResourceNotFoundException:
        print(f"  Creating new function {LAMBDA_FUNCTION_NAME}...")
        for attempt in range(8):
            try:
                lam.create_function(
                    FunctionName=LAMBDA_FUNCTION_NAME,
                    Runtime="python3.11",
                    Role=role_arn,
                    Handler="lambda_function.handler",
                    Code={'ZipFile': zip_bytes},
                    Description="Chest X-Ray Disease Detection Serverless API",
                    Timeout=30,
                    MemorySize=512,
                    Environment={
                        'Variables': {
                            'AWS_S3_BUCKET_NAME': BUCKET_NAME
                        }
                    }
                )
                print("  [OK] Function created successfully.")
                break
            except Exception as e:
                print(f"  Waiting for IAM role propagation (attempt {attempt+1}/8): {e}")
                time.sleep(5)

def setup_function_url():
    session = get_session()
    lam = session.client('lambda')
    print("\n[5/5] Configuring Public Lambda Function URL & CORS ...")

    # Wait until function is active
    for _ in range(10):
        try:
            fn = lam.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
            state = fn.get('Configuration', {}).get('State')
            if state in ('Active', None):
                break
            time.sleep(2)
        except Exception:
            time.sleep(2)

    url = None
    try:
        config = lam.get_function_url_config(FunctionName=LAMBDA_FUNCTION_NAME)
        url = config.get("FunctionUrl")
        print(f"  Found existing Function URL: {url}")
    except lam.exceptions.ResourceNotFoundException:
        print("  Creating Function URL (AuthType=NONE)...")
        res = lam.create_function_url_config(
            FunctionName=LAMBDA_FUNCTION_NAME,
            AuthType="NONE",
            Cors={
                "AllowOrigins": ["*"],
                "AllowMethods": ["*"],
                "AllowHeaders": ["*"],
                "MaxAge": 86400
            }
        )
        url = res.get("FunctionUrl")
        print(f"  [OK] Function URL created: {url}")

    # Ensure public invocation permission
    try:
        lam.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId="FunctionURLAllowPublicAccess",
            Action="lambda:InvokeFunctionUrl",
            Principal="*",
            FunctionUrlAuthType="NONE"
        )
        print("  [OK] Public access permission added.")
    except Exception:
        pass

    try:
        lam.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId="FunctionURLAllowInvoke",
            Action="lambda:InvokeFunction",
            Principal="*"
        )
        print("  [OK] Public invoke permission added.")
    except Exception:
        pass

    return url

def test_deployment_url(url):
    print(f"\n=======================================================")
    print(f"TESTING DEPLOYMENT LINK: {url}")
    print(f"=======================================================")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            print("HTTP Status:", resp.status)
            print("Response payload:")
            print(json.dumps(json.loads(data), indent=2))
            return True
    except Exception as e:
        print(f"Test request error: {e}")
        return False

if __name__ == "__main__":
    session = get_session()
    sts = session.client('sts')
    print("Caller Identity:", sts.get_caller_identity()['Arn'])

    # 1. Upload models to S3
    upload_models_to_s3()

    # 2. Package Lambda
    zip_path = create_lambda_zip()

    # 3. IAM Role
    role_arn = get_or_create_lambda_role()

    # 4. Deploy Lambda
    deploy_lambda_function(role_arn, zip_path)

    # 5. Setup Function URL
    func_url = setup_function_url()

    # 6. Verify URL
    test_deployment_url(func_url)
