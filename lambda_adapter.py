import os
from mangum import Mangum
from backend.main import app

# Disable OpenAPI docs redirect issues on serverless
handler = Mangum(app, lifespan="off")

